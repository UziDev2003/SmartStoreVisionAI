"""Smart Store Vision AI - Streamlit App"""
import streamlit as st, cv2, time, tempfile
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="Smart Store Vision AI", page_icon="🏪", layout="wide")
st.markdown("""<style>
.stMetric>div:first-child{color:#00d9ff!important;font-size:1.5rem!important}
.stMetric>div:nth-child(2){color:#ffffff!important}
.zone{background:#0f3460;color:#ffffff;padding:10px;border-radius:8px;margin:5px 0}
</style>""", unsafe_allow_html=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from src.detector.yolo_detector import YOLODetector
from src.tracker.bytetrack import ByteTracker
from src.analytics.zone_analytics import ZoneAnalytics, Zone
from src.analytics.heatmap import HeatmapGenerator
from src.analytics.behavior_detector import BehaviorDetector
from src.alerts.alert_manager import AlertManager

@st.cache_resource
def get_det(ms='s', cf=0.5, dv='cpu'): return YOLODetector(model_size=ms, confidence_threshold=cf, device=dv)

def mk_zones():
    return [Zone("e","Entrance","entrance",[(100,400),(300,400),(300,600),(100,600)],10),
            Zone("c","Checkout","checkout",[(500,400),(700,400),(700,600),(500,600)],8),
            Zone("r","Restricted","restricted",[(800,100),(950,100),(950,250),(800,250)],0)]

for k,v in {'run':False,'st':[],'am':AlertManager(),'hm':None,'zA':None,'zones':None}.items():
    if k not in st.session_state: st.session_state[k]=v

st.markdown("## 🏪 Smart Store Vision AI")
st.caption("YOLOv8 + ByteTrack Analytics")

with st.sidebar:
    st.header("⚙️ Settings")
    ms=st.selectbox("Model",["n","s","m"],1)
    cf=st.slider("Confidence",0.2,0.9,0.5,0.05)
    dv=st.selectbox("Device",["cpu","cuda"],0)
    loiter=st.slider("Loiter (s)",30,300,120,10)
    showHM=st.checkbox("Show Heatmap",False)

c1,c2=st.columns([3,1])

with c2:
    st.subheader("📊 Stats")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        fps_metric = st.empty()
        footfall_metric = st.empty()
    with metric_col2:
        tracks_metric = st.empty()
        alerts_metric = st.empty()
    
    st.subheader("📍 Zones")
    zones_container = st.empty()
    
    st.subheader("⚠️ Alerts")
    alerts_container = st.empty()
    
    st.subheader("🗺️ Heatmap")
    heatmap_container = st.empty()

    # Pre-populate placeholders with current session state values
    if st.session_state.st:
        s = st.session_state.st[-1]
        fps_metric.metric("FPS", f"{s['fps']:.1f}")
        tracks_metric.metric("Tracks", s['tr'])
        footfall_metric.metric("Footfall", s['fh'])
    else:
        fps_metric.metric("FPS", "0.0")
        tracks_metric.metric("Tracks", "0")
        footfall_metric.metric("Footfall", "0")
        
    alerts_metric.metric("Alerts", len(st.session_state.am.get_recent(60)))
    
    if st.session_state.zA and st.session_state.zones:
        with zones_container.container():
            for zid, st_ in st.session_state.zA.stats.items():
                z = next((zz for zz in st.session_state.zones if zz.zone_id == zid), None)
                if z:
                    c = "🔴" if z.zone_type == "restricted" else "🟡"
                    st.markdown(f'<div class="zone">{c} {z.name}: {st_.current_occupancy}</div>', unsafe_allow_html=True)
    else:
        zones_container.info("No active zones")
        
    recent_alerts = st.session_state.am.get_recent(60)
    if recent_alerts:
        with alerts_container.container():
            for a in recent_alerts[:4]:
                st.markdown(f'**{datetime.fromtimestamp(a.timestamp).strftime("%H:%M:%S")}** {a.message}')
    else:
        alerts_container.info("No recent alerts")
        
    if st.session_state.hm:
        heatmap_container.image(cv2.cvtColor(st.session_state.hm.get_image(), cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        heatmap_container.info("No heatmap data")

with c1:
    st.subheader("📹 Upload Video")
    up=st.file_uploader("Choose video",type=['mp4','avi','mov'],label_visibility="collapsed")
    if up:
        tf=tempfile.NamedTemporaryFile(delete=False,suffix='.mp4')
        tf.write(up.read());tf.close()
        b1,b2=st.columns(2)
        with b1:
            if st.button("▶️ Start",type="primary",use_container_width=True): st.session_state.run=True
        with b2:
            if st.button("⏹️ Stop",use_container_width=True): st.session_state.run=False
        fp=st.empty();bar=st.progress(0)
        if st.session_state.run:
            try: det=get_det(ms,cf,dv)
            except Exception as e: st.error(f"Model error: {e}");det=None
            if det:
                st.session_state.st = [] # Reset stats history
                st.session_state.zones = mk_zones()
                st.session_state.zA = ZoneAnalytics(st.session_state.zones, 960, 540)
                st.session_state.hm = HeatmapGenerator(960, 540)
                st.session_state.am = AlertManager()
                
                trk=ByteTracker()
                bhv=BehaviorDetector(loiter_threshold=loiter)
                cap=cv2.VideoCapture(tf.name);fc=0
                while st.session_state.run:
                    ret,fr=cap.read()
                    if not ret:cap.set(cv2.CAP_PROP_POS_FRAMES,0);continue
                    fc+=1;fr=cv2.resize(fr,(960,540));ts=time.time()
                    dets=det.detect(fr);res=trk.update(fr,dets,ts)
                    st.session_state.zA.update(res.tracks,ts)
                    st.session_state.hm.update(res.tracks,ts)
                    cfg={z.zone_id:{"name":z.name,"dwell_threshold":z.dwell_threshold} for z in st.session_state.zones}
                    for a in bhv.detect(res.tracks,cfg,st.session_state.zA.zone_occ,ts):
                        t=next((tr for tr in res.tracks if tr.track_id==a.track_id),None)
                        st.session_state.am.create(a.alert_type.value,"v1",a.track_id,a.message,a.severity.value,a.zone_id,fr,t.bbox if t else None)
                    out=det.draw_detections(fr.copy(),dets);out=trk.draw_tracks(out,res.tracks);out=st.session_state.zA.draw_zones(out)
                    if showHM:out=st.session_state.hm.get_overlay(out,0.3)
                    s=det.get_performance_stats()
                    cv2.putText(out,f"FPS:{s['fps']:.0f} T:{len(res.tracks)} F:{st.session_state.hm.total_footfall}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                    fp.image(cv2.cvtColor(out,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
                    bar.progress(min(fc/100,1.0))
                    
                    st.session_state.st.append({'fps':s['fps'],'tr':len(res.tracks),'fh':st.session_state.hm.total_footfall})
                    
                    # Update placeholders in real-time
                    fps_metric.metric("FPS", f"{s['fps']:.1f}")
                    tracks_metric.metric("Tracks", len(res.tracks))
                    footfall_metric.metric("Footfall", st.session_state.hm.total_footfall)
                    alerts_metric.metric("Alerts", len(st.session_state.am.get_recent(60)))
                    
                    with zones_container.container():
                        for zid,st_ in st.session_state.zA.stats.items():
                            z=next((zz for zz in st.session_state.zones if zz.zone_id==zid),None)
                            if z:
                                c="🔴" if z.zone_type=="restricted" else "🟡"
                                st.markdown(f'<div class="zone">{c} {z.name}: {st_.current_occupancy}</div>',unsafe_allow_html=True)
                                
                    curr_alerts = st.session_state.am.get_recent(60)
                    if curr_alerts:
                        with alerts_container.container():
                            for a in curr_alerts[:4]:
                                st.markdown(f'**{datetime.fromtimestamp(a.timestamp).strftime("%H:%M:%S")}** {a.message}')
                    else:
                        alerts_container.info("No recent alerts")
                        
                    heatmap_container.image(cv2.cvtColor(st.session_state.hm.get_image(),cv2.COLOR_BGR2RGB),use_container_width=True)
                    
                    time.sleep(0.03)
                cap.release();bar.empty();st.session_state.run=False
    else: st.info("👆 Upload video to start")
