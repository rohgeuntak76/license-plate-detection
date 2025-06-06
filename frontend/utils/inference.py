import requests
import json
import io
import websocket
import streamlit as st
import pandas as pd
from utils.logging import logger  
import time

def display_frame(frame_bytes,ann_frame,results_frame):
    logger.info("display frame function called")
    # print(type(frame_bytes))
    # image = io.BytesIO(frame_bytes) # Convert bytes to image
    # results_frame.append(image)
    # ann_frame.image(image) # Update the placeholder
    results_frame.append(frame_bytes)
    ann_frame.image(frame_bytes) # Update the placeholder
    logger.info("display frame Done!")

def display_dataframe(detection_results,results_list,result_df):
# def display_dataframe(detection_results,results_list):
    logger.info("display dataframe function called")
    results_json = json.loads(detection_results)
    result_df.add_rows(results_json)
    logger.info(f"{results_json}")
    results_list.extend(results_json) # update list
    logger.info("display dataframe Done!")

def vehicle_detection_image(api_host, selected_classes, vid_file,vehicle_conf,selected_ind):
    if selected_classes[0] == 'License_Plate': # license plate detection usecase
        url = "http://" + api_host + "/api/image/plates/detect/annotated"
    else: # vehicle detction usecase
        url = "http://" + api_host + "/api/image/vehicles/detect/annotated"
    # Do inference
    files = {
            'image': (vid_file.name, vid_file.getvalue(), 'image/jpeg'),
        }
    data = {
        'conf': f'{vehicle_conf}'
    }
    params = [('classes', str(n)) for n in selected_ind]

    response = requests.post(url,files=files,data=data,params=params,stream=True)
    annotated_result = io.BytesIO(response.content)
    return annotated_result


def vehicle_detection_video_file(api_host,sess_id,video_path,output_path,selected_classes,vehicle_conf,selected_ind,ann_frame,video_inference_ratio):
    if selected_classes[0] == 'License_Plate':
        url = f"http://{api_host}/api/video/plates/detect/annotatedVideo"
    else:
        url = f"http://{api_host}/api/video/vehicles/detect/annotatedVideo"
    
    payload = {
        'video_path' : f'{video_path}',
        'output_path' : f'{output_path}',
        'conf': f'{vehicle_conf}',
        'ratio': f'{video_inference_ratio}',
    }
    
    params = [('classes', str(n)) for n in selected_ind]
    response = requests.post(url,json=payload,params=params,stream=True)
    if response.status_code == 200:
        logger.info("Video download Done!!")
        annotated_result = io.BytesIO(response.content)
        return annotated_result
    else:
        return


def vehicle_detection_video(api_host,sess_id,video_path,selected_classes,vehicle_conf,selected_ind,ann_frame,video_inference_ratio):
    # import websocket
    if selected_classes[0] == 'License_Plate':
        ws_url = f"ws://{api_host}/api/video/ws/license_plate/"
    else:
        ws_url = f"ws://{api_host}/api/video/ws/vehicle/"
    ws_url = ws_url + sess_id
    results_frame = []
    ws = websocket.WebSocketApp(
        ws_url,
        on_message=lambda ws, msg: display_frame(msg,ann_frame,results_frame),
        on_error=lambda ws, err: st.error(f"Error: {err}"),
        on_close=lambda ws,status,msg: st.info(f"Processing complete. code {status} : {msg}")
    )

    # Send video path to backend
    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"conf":vehicle_conf,"classes":selected_ind,"ratio":video_inference_ratio}))
    with st.spinner("Wait for Inferencing...", show_time=True):
        ws.run_forever()
    for frame in results_frame:
        ann_frame.image(frame)
        time.sleep(0.1)

def license_number_image_infer(api_host,vid_file,vehicle_conf,license_conf):
    files = {
        'image': (vid_file.name, vid_file.getvalue(), 'image/jpeg'),
    }
    data = {
        'vehicle_conf': f'{vehicle_conf}',
        'license_conf': f'{license_conf}',
    }
    
    url = "http://" + api_host + "/api/image/plate_number/crop/detect/info"
    response_info = requests.post(url,files=files,data=data)
    response_info_json = response_info.json()
    return response_info_json

def license_number_image_visualize(api_host,vid_file,detection_result):
    files = {
        'image': (vid_file.name, vid_file.getvalue(), 'image/jpeg'),
    }
    data = {
        'item_json': json.dumps(detection_result),
    }

    url = 'http://' + api_host + '/api/utils/draw/annotated'
    response = requests.post(url,files=files,data=data,stream=True)
    annotated_result = io.BytesIO(response.content)
    return annotated_result

def license_number_video_infer(api_host,sess_id,video_path,vehicle_conf,license_conf,video_inference_ratio):
    results_list = []
    result_df = st.dataframe(pd.DataFrame(columns=("frame_number","track_id","vehicle_bbox","vehicle_bbox_score","lp_bbox","lp_bbox_score","lp_number","lp_text_score")),hide_index=True)
    # streaming_dataframe(results_list)
    ws = websocket.WebSocketApp(
        f"ws://{api_host}/api/video/ws/license_number/{sess_id}",
        on_message=lambda ws, msg: display_dataframe(msg,results_list,result_df),
        # on_message=lambda ws, msg: display_dataframe(msg,results_list),
        on_error=lambda ws, err: st.error(f"Error: {err}"),
        on_close=lambda ws,status,msg: st.info(f"Processing complete. code {status} : {msg}")
    )
    # Send video path to backend
    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"vehicle_conf": vehicle_conf,"license_conf":license_conf,"ratio":video_inference_ratio}))
    ws.run_forever()
    return results_list


def license_number_video_visualize(api_host,sess_id,video_path,detection_result,ann_frame):
    results_frame = []
    ws = websocket.WebSocketApp(
        f"ws://{api_host}/api/utils/ws/draw/annotated/{sess_id}",
        on_message=lambda ws,msg: display_frame(msg,ann_frame,results_frame),
        on_error=lambda ws,err: st.error(f"Error: {err}"),
        on_close=lambda ws,status,msg: st.info(f"Processing complete. code {status} : {msg}")
    )

    # Send video path to backend
    ws.on_open = lambda ws: ws.send(json.dumps({"video_path": video_path,"detection_results":detection_result}))
    with st.spinner("Wait for Visualization...", show_time=True):
        ws.run_forever()

    for frame in results_frame:
        ann_frame.image(frame)
        time.sleep(0.1)

def license_number_video_visualize_file(api_host,video_path,output_path,detection_result):
    url = f"http://{api_host}/api/utils/draw/annotatedVideo"
    
    payload = {
        'video_path' : video_path,
        'output_path' : output_path,
        'detection_result': detection_result,
    }
    
    response = requests.post(url,json=payload,stream=True)
    if response.status_code == 200:
        logger.info("Video download Done!!")
        annotated_result = io.BytesIO(response.content)
        return annotated_result
    else:
        return
