"""Validate the actual PhiView browser against a running H200 service."""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import sync_playwright

ap=argparse.ArgumentParser();ap.add_argument('--url',required=True);ap.add_argument('--out',required=True)
a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1600,'height':1000},device_scale_factor=1)
 errors=[];requests=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:requests.append(r.url))
 page.goto(a.url,wait_until='domcontentloaded')
 page.wait_for_function('document.getElementById("frame").naturalWidth > 0')
 page.wait_for_function('document.querySelectorAll("#objects button").length === 18')
 def cmd(op,**kw):
  r=page.request.post(a.url+'/api/command',data={'op':op,**kw});assert r.ok,r.text();return r.json()
 cmd('view',mode='original');cmd('camera');page.wait_for_timeout(700)
 page.screenshot(path=str(out/'browser-original.png'))
 box=page.locator('#frame').bounding_box();assert box
 page.mouse.click(box['x']+box['width']*.33,box['y']+box['height']*.42)
 page.wait_for_timeout(900)
 selected=page.request.get(a.url+'/api/status').json()['selected'];assert selected is not None,'Image click did not select an object'
 page.screenshot(path=str(out/'browser-selected.png'))
 # The keyboard should change server-rendered frames and release cleanly.
 fid=page.request.get(a.url+'/api/status').json()['frame']
 page.locator('#viewport').focus();page.keyboard.down('w');page.wait_for_timeout(220);page.keyboard.up('w')
 page.wait_for_timeout(400)
 moved=page.request.get(a.url+'/api/status').json()['frame'];assert moved>fid
 page.mouse.move(box['x']+box['width']*.5,box['y']+box['height']*.5)
 page.mouse.down(button='right');page.mouse.move(box['x']+box['width']*.56,box['y']+box['height']*.52,steps=6);page.mouse.up(button='right')
 page.wait_for_timeout(500)
 page.screenshot(path=str(out/'browser-navigation.png'))
 # Actual controls, including simulation state transitions.
 page.locator('#enable').click();page.wait_for_timeout(500)
 assert page.request.get(a.url+'/api/status').json()['mode']=='simulation'
 page.locator('#fall').click();page.wait_for_timeout(900)
 s=page.request.get(a.url+'/api/status').json();assert s['running'] and s['sim_time']>0
 page.locator('#pause').click();page.wait_for_timeout(400)
 assert not page.request.get(a.url+'/api/status').json()['running']
 page.locator('#reset').click();page.locator('#view').select_option('clean_selected');page.wait_for_timeout(600)
 assert page.request.get(a.url+'/api/status').json()['mode']=='clean_selected'
 page.screenshot(path=str(out/'browser-clean-selected.png'))
 cmd('view',mode='original');cmd('camera');page.wait_for_timeout(500)
 assert not errors,errors
 assert page.locator('canvas').count()==0
 assert not any(x.split('?')[0].endswith(('.ply','.glb','.splat','.obj')) for x in requests)
 report={'passed':True,'selected_by_image_click':selected,'keyboard_frames':[fid,moved],
         'page_errors':errors,'request_paths':sorted(set(x.removeprefix(a.url).split('?')[0] for x in requests)),
         'canvas_count':0,'objects':18,'gpu':s['gpu']}
 (out/'browser-check.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
 browser.close()
