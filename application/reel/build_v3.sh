set -e
cd /home/user/w3
mkdir -p fonts
curl -sSfL -o fonts/m.ttf https://raw.githubusercontent.com/JetBrains/JetBrainsMono/master/fonts/ttf/JetBrainsMono-Regular.ttf
G=https://raw.githubusercontent.com/kruuusher13
curl -sSf -o sat.mp4 "$SAT"; curl -sSf -o rov.mp4 "$ROV"; curl -sSf -o hall.mp4 "$HALL"
curl -sSf -o m2.mp4 "$G/BlueRov-Object-Avoidance/main/docs/Depth_media/My%20Movie%202.mp4"
curl -sSf -o mon.png $G/gptnl-energy-estimation-ekf/main/assets/monitor_live.png
npm i --silent katex @fontsource/spectral @fontsource/archivo > /dev/null 2>&1
export NODE_PATH=$(pwd)/node_modules:$(npm root -g)
node capture.js
node layers.js
python3 shots3.py
python3 music3.py
ffmpeg -v error -y -i base.mp4 -i music.wav -filter_complex "[0:v]fade=t=in:st=0:d=0.5,fade=t=out:st=29.4:d=0.6,format=yuv420p[v];[1:a]afade=t=out:st=28.8:d=1.2[a]" -map "[v]" -map "[a]" -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart -t 30 final.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 final.mp4
python3 - <<'PY'
import subprocess
from PIL import Image
ts=[1.4,4.0,6.5,8.5,10.8,13.0,15.5,18.9,22.0,24.6,26.2,29.0]
ims=[]
for t in ts:
    subprocess.run(["ffmpeg","-v","error","-y","-ss",str(t),"-i","final.mp4","-frames:v","1","-vf","scale=480:-2","fr.png"]); ims.append(Image.open("fr.png").convert("RGB"))
g=Image.new("RGB",(960,270*6))
for i,im in enumerate(ims): g.paste(im,((i%2)*480,(i//2)*270))
g.save("grid3.jpg",quality=72)
PY
[ -n "$UP" ] && curl -sS -f -X PUT -H "Content-Type: video/mp4" -H "If-None-Match: *" --upload-file final.mp4 "$UP" -o /dev/null -w "UPLOAD %{http_code}\n"
echo DONE
