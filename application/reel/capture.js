const { chromium } = require('playwright');
const fs=require('fs');
const BASE=process.env.BASE||'https://kruuusher13.github.io/Portfolio/';
const EXE=process.env.CHROME||undefined;
const out='cap'; fs.mkdirSync(out,{recursive:true});
const pad=n=>String(n).padStart(3,'0');
(async()=>{
 const b=await chromium.launch(EXE?{executablePath:EXE}:{});
 // ---- graph hovers
 let ctx=await b.newContext({viewport:{width:1920,height:1080},deviceScaleFactor:2});
 let p=await ctx.newPage(); await p.goto(BASE,{waitUntil:'networkidle'}); await p.waitForTimeout(1500);
 const clip={x:340,y:195,width:1240,height:860};
 await p.mouse.move(5,5); await p.waitForTimeout(400);
 await p.screenshot({path:`${out}/g_none.png`,clip});
 for (const t of ['ISRO SAC','CSIR NIO','Torxflow','ConcertLab','TNO','Amdahl bound','Extended Kalman','Masked classification','Potential fields']){
   try{ const l=p.getByText(t,{exact:true}).first(); await l.hover({timeout:3000}); await p.waitForTimeout(350);
        await p.screenshot({path:`${out}/g_${t.replace(/ /g,'_')}.png`,clip}); }catch(e){console.log('hover fail',t)}
 }
 await ctx.close();
 // ---- method figures
 ctx=await b.newContext({viewport:{width:1920,height:1080},deviceScaleFactor:3});
 p=await ctx.newPage(); await p.goto(BASE,{waitUntil:'networkidle'}); await p.waitForTimeout(1200);
 for (const x of await p.$$('button')) if((await x.innerText()).toLowerCase().includes('method')) await x.click();
 await p.waitForTimeout(1200);
 await p.evaluate(()=>{const f=[...document.querySelectorAll('figure')];
   const pairs=[[0,1],[4,5],[6,7],[8,9]];
   pairs.forEach(([a,c],k)=>{let el=f[a].parentElement; while(el && !el.contains(f[c])) el=el.parentElement; el.id='cap'+k; el.style.background='#fff'; el.style.padding='14px';});
   window.setRange=(el,v)=>{const s=Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set; s.call(el,String(v)); el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true}));};
 });
 const boxes=await p.evaluate(()=>{const o={}; for(let k=0;k<4;k++){const c=document.getElementById('cap'+k); const cr=c.getBoundingClientRect();
   const svgs=[...c.querySelectorAll('svg')].map(s=>s.getBoundingClientRect()).filter(r=>r.width>200).sort((a,b)=>b.width*b.height-a.width*a.height);
   const u=svgs.slice(0,(k==3||k==2)?2:1); const x0=Math.min(...u.map(r=>r.left)),y0=Math.min(...u.map(r=>r.top)),x1=Math.max(...u.map(r=>r.right)),y1=Math.max(...u.map(r=>r.bottom));
   o['cap'+k]=[x0-cr.left,y0-cr.top,x1-x0,y1-y0].map(v=>Math.round(v*3));} return o;});
 fs.writeFileSync(`${out}/boxes.json`,JSON.stringify(boxes)); console.log(JSON.stringify(boxes));
 async function sweep(k,ri,steps,name,pre){
   const loc=p.locator('#cap'+k); await loc.scrollIntoViewIfNeeded();
   if(pre) await pre();
   const r=await p.evaluate(([k,ri])=>{const e=document.querySelectorAll('#cap'+k+' input[type=range]')[ri]; return e?[+e.min,+e.max,+e.value,+(e.step||1)]:null},[k,ri]);
   console.log(name,r);
   for(let i=0;i<steps;i++){
     if(r){ const [mn,mx]=r; const v=mx-(mx-mn)*(i/(steps-1)); await p.evaluate(([k,ri,v])=>{window.setRange(document.querySelectorAll('#cap'+k+' input[type=range]')[ri],v)},[k,ri,v]); }
     await p.waitForTimeout(60); await loc.screenshot({path:`${out}/${name}_${pad(i)}.png`});
   }
 }
 const clickBtn=async(k,txt)=>{await p.locator('#cap'+k+' button',{hasText:txt}).first().click(); await p.waitForTimeout(250);};
 await sweep(0,0,40,'energy');
 await sweep(1,0,40,'pf',async()=>{await clickBtn(1,'Obstacle field');});
 // chess stepper
 const loc2=p.locator('#cap2'); await loc2.scrollIntoViewIfNeeded(); let i=0;
 for (const s of ['Position','Embed','Encoder','CLS','Move head','Legality mask']){ await clickBtn(2,s); await loc2.screenshot({path:`${out}/chess_${pad(i++)}.png`}); }
 // fpga
 await sweep(3,0,1,'fpgaA',async()=>{await clickBtn(3,'Alternating');});
 await sweep(3,0,1,'fpgaB',async()=>{await clickBtn(3,'Overlapped');});
 await ctx.close(); await b.close(); console.log('capture done');
})();
