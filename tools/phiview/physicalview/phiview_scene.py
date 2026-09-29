"""Full Gaussian scene and visibility-aware picking; all arrays stay on the server."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from physicalview.render import look_at_w2c


@dataclass
class FlyCamera:
    position: np.ndarray
    yaw: float
    pitch: float
    fov: float = 60.0
    speed: float = 1.2
    orbit_distance: float | None = None

    @classmethod
    def from_w2c(cls, w2c, fov=60.0):
        c = np.linalg.inv(w2c)
        f = c[:3, 2]
        return cls(c[:3, 3].copy(), float(np.arctan2(f[1], f[0])),
                   float(np.arcsin(np.clip(f[2], -1, 1))), fov)

    def forward(self):
        return np.array([np.cos(self.pitch)*np.cos(self.yaw),
                         np.cos(self.pitch)*np.sin(self.yaw), np.sin(self.pitch)])

    def update(self, keys, dt, look=(0, 0), boost=False):
        self.yaw -= float(np.clip(look[0], -2000, 2000)) * .0025
        self.pitch = float(np.clip(self.pitch-float(np.clip(look[1], -2000, 2000))*.0025,
                                   -np.pi/2+.02, np.pi/2-.02))
        right = np.array([np.sin(self.yaw), -np.cos(self.yaw), 0.])
        v = self.forward() * (int('w' in keys)-int('s' in keys))
        v += right * (int('d' in keys)-int('a' in keys))
        v[2] += int('e' in keys)-int('q' in keys)
        norm = np.linalg.norm(v)
        if norm:
            self.position += v/norm * self.speed * (3 if boost else 1) * min(max(dt, 0), .1)

    def matrices(self, wh):
        w, h = wh
        f = h / (2*np.tan(np.deg2rad(self.fov)/2))
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1.]])
        return look_at_w2c(self.position, self.position+self.forward()), K


def subset(gs, idx):
    return {k: gs[k][idx] if k != 'sh_degree' else gs[k] for k in gs}


class GaussianScene:
    def __init__(self, state):
        import torch
        from agents.core.common import load_gaussians
        self.state = state
        self.raw = state.splat_gs
        self.rasterize_mode = 'antialiased'
        if self.raw is None:
            raise RuntimeError('Original Gaussian scene is required')
        self.names = list(sorted(state.objects))
        self.ids = {n: i+1 for i, n in enumerate(self.names)}
        self.labels = torch.zeros(len(self.raw['means']), dtype=torch.long, device='cuda')
        self.mask_sources = {}
        self.indices = {}
        for n in self.names:
            path = state.result_set.out_dir/'inpaint'/n/'removal_idx.npy'
            if state.objects[n].meta.get('interactive'):
                path = state.objects[n].dir/'gaussian_indices.npy'
            if path.exists():
                idx = np.asarray(np.load(path), dtype=np.int64)
                if idx.ndim != 1 or (idx.size and (idx.min() < 0 or idx.max() >= len(self.labels))):
                    raise ValueError(f'Invalid Gaussian mask: {path}')
                self.mask_sources[n] = state.objects[n].meta.get('mask_source', 'cached removal mask')
                idx = torch.as_tensor(idx, device='cuda')
            else:
                box = state.objects[n].meta.get('aabb')
                if box is None:
                    idx = torch.empty(0, dtype=torch.long, device='cuda')
                    self.mask_sources[n] = 'unavailable'
                else:
                    lo, hi = torch.tensor(box, device='cuda')
                    idx = torch.where(((self.raw['means'] >= lo-.015) &
                                       (self.raw['means'] <= hi+.015)).all(-1))[0]
                    self.mask_sources[n] = 'proposal bounding region (approximate)'
            self.indices[n] = idx
            self.labels[idx] = self.ids[n]
        self.clean = state.clean_bg_gs
        self.canonical = {}
        self.variants = {n: 'original' for n in self.names}
        union = state.result_set.out_dir/'inpaint'/'removal_union_idx.npy'
        self.removed = torch.zeros(len(self.labels), dtype=torch.bool, device='cuda')
        if union.exists():
            self.removed[torch.as_tensor(np.load(union).astype(np.int64), device='cuda')] = True
        else:
            self.removed = self.labels > 0
        if any(rec.meta.get('interactive') for rec in state.objects.values()):
            self.removed |= self.labels > 0
            self.clean = None
        self.count = len(self.labels)
        self.prompt_backgrounds = {}

    def add_object(self, name, indices, source, supersedes=()):
        if name in self.ids:
            raise ValueError('Object already registered')
        for old in supersedes:
            self.labels[self.indices[old]] = 0
            self.names.remove(old)
            for mapping in (self.ids, self.indices, self.mask_sources, self.variants):
                mapping.pop(old)
        if supersedes:
            for i, old in enumerate(self.names, 1):
                self.ids[old] = i
                self.labels[self.indices[old]] = i
        self.names.append(name)
        self.ids[name] = len(self.names)
        self.indices[name] = indices
        self.labels[indices] = self.ids[name]
        self.removed[indices] = True
        self.mask_sources[name] = source
        self.variants[name] = 'original'
        # Previous background products did not remove this newly discovered object.
        self.clean = None
        self.state.clean_bg_gs = None
        self.prompt_backgrounds.clear()

    def choose(self, name, source):
        if source == 'original':
            self.variants[name] = source
            return
        from physicalview.scene_state import object_canonical_gs
        from agents.core.common import pad_sh, transform_gaussians
        rec = self.state.objects[name]
        prop = rec.proposals.get(source)
        if prop is None or prop.aligned is None or not prop.aligned.get('T') or prop.aligned.get('rejected'):
            raise ValueError('This alternative needs valid registration before it can be used')
        gs = object_canonical_gs(self.state, name, source)
        gs = {k: v.to('cuda') if hasattr(v, 'to') else v for k, v in gs.items()}
        gs = pad_sh(gs, self.raw['sh_degree'])
        self.canonical[(name, source)] = transform_gaussians(gs, np.asarray(prop.aligned['T']))
        self.variants[name] = source

    def compose(self, mode, selected, simulatable, transforms=None):
        import torch
        from agents.core.common import cat_gaussians, transform_gaussians
        transforms = transforms or {}
        if mode == 'original':
            return self.raw, self.labels
        if self.clean is None and mode != 'simulation':
            raise ValueError('An inpainted background is required for this view')
        hidden = set(simulatable) if mode == 'clean_all' else ({selected} if mode == 'clean_selected' else set())
        prompt_bg = self.prompt_backgrounds.get(frozenset(hidden)) if hidden else None
        if prompt_bg is not None:
            # Prompt products already contain the untouched original objects.
            # Object labels are rendered separately from their original Gaussians below
            # only for picking in the default clean composition; here the clean image is
            # deliberately unlabelled until masks are regenerated for that version.
            return prompt_bg, torch.zeros(len(prompt_bg['means']), device='cuda', dtype=torch.long)
        if self.clean is None:
            # Simulation can reveal unobserved regions, without claiming completion.
            # Keep the original observed background and move only enabled objects.
            background = ~self.removed
            parts, labels = [subset(self.raw, background)], [self.labels[background]]
        else:
            parts, labels = [self.clean], [torch.zeros(len(self.clean['means']), device=self.labels.device, dtype=torch.long)]
        # Restore original carved Gaussians except hidden or replaced objects. This preserves
        # original appearance for all unselected objects, including rejected proposals.
        restore = self.removed.clone()
        for n in self.names:
            if n in hidden or n in transforms or self.variants[n] != 'original':
                restore[self.indices[n]] = False
        parts.append(subset(self.raw, restore)); labels.append(self.labels[restore])
        for n in self.names:
            if n in hidden:
                continue
            source = self.variants[n]
            if n not in transforms and source == 'original':
                continue
            gs = subset(self.raw, self.indices[n]) if source == 'original' else self.canonical[(n, source)]
            if n in transforms:
                gs = transform_gaussians(gs, transforms[n])
            parts.append(gs)
            labels.append(torch.full((len(gs['means']),), self.ids[n], device=self.labels.device, dtype=torch.long))
        return cat_gaussians(parts), torch.cat(labels)

    def render(self, camera, wh, mode, selected, simulatable, transforms=None, highlight=True):
        import torch
        from gsplat import rasterization
        gs, labels = self.compose(mode, selected, simulatable, transforms)
        w2c, K = camera.matrices(wh)
        # Rasterize object membership against the SAME occluding scene geometry.
        with torch.inference_mode():
            viewmats=torch.tensor(w2c,dtype=torch.float32,device='cuda')[None]
            intrinsics=torch.tensor(K,dtype=torch.float32,device='cuda')[None]
            raster_args=dict(means=gs['means'],quats=gs['quats'],scales=gs['scales'],
                             opacities=gs['opacities'],viewmats=viewmats,Ks=intrinsics,
                             width=wh[0],height=wh[1],packed=False,near_plane=.01,far_plane=100.,
                             rasterize_mode=self.rasterize_mode)
            # Screen-space opacity compensation belongs to rasterization; original
            # Gaussian rows/parameters remain intact. Use it for RGB and picking alike.
            color,_,_=rasterization(**raster_args,colors=gs['sh'],sh_degree=gs['sh_degree'],render_mode='RGB+ED')
            rgb=color[0,...,:3].clamp(0,1).cpu().numpy()
            depth=color[0,...,3].cpu().numpy()
            features = torch.nn.functional.one_hot(labels, len(self.names)+1).float()
            out, _, _ = rasterization(**raster_args,colors=features,sh_degree=None)
            confidence, ids = out[0].max(-1)
            ids[confidence < .25] = 0
            mask = ids.cpu().numpy().astype(np.uint16)
        self.last_rgb = (np.clip(rgb, 0, 1)*255+.5).astype(np.uint8)
        if highlight or selected is not None:
            from physicalview.highlight import highlight_objects
            rgb = highlight_objects(rgb, mask, self.ids.get(selected), show_all=highlight)
        return (np.clip(rgb, 0, 1)*255+.5).astype(np.uint8), mask, depth, w2c, K
