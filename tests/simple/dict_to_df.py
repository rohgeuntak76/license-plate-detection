import pandas as pd

value = {1.0: {'car': {'bbox': [678.2144775390625, 1232.2449951171875, 1315.8944091796875, 1791.5057373046875], 'bbox_score': 0.894150972366333}, 'license_plate': {'bbox': [221.50367736816406, 379.7851257324219, 392.10955810546875, 429.76385498046875], 'bbox_score': 0.6572830677032471, 'number': None, 'text_score': None}}, 4.0: {'car': {'bbox': [1956.56298828125, 1046.917724609375, 2509.57958984375, 1566.6651611328125], 'bbox_score': 0.8615961074829102}, 'license_plate': {'bbox': [222.3783721923828, 373.5968017578125, 370.99041748046875, 421.90069580078125], 'bbox_score': 0.7343707084655762, 'number': 'MU51TSU', 'text_score': 0.035259282382269844}}}
# value_flatten = {'track_id': 1.0,'car_bbox': [678.2144775390625, 1232.2449951171875, 1315.8944091796875, 1791.5057373046875],'car_bbox_score': 0.894150972366333,'lp_bbox': [221.50367736816406, 379.7851257324219, 392.10955810546875, 429.76385498046875], 'lp_bbox_score': 0.6572830677032471, 'lp_number': None, 'lp_text_score': None}
value_flatten = {
    1.0: {

        'car_bbox': [678.2144775390625, 1232.2449951171875, 1315.8944091796875, 1791.5057373046875],
        'car_bbox_score': 0.894150972366333,
        'lp_bbox': [221.50367736816406, 379.7851257324219, 392.10955810546875, 429.76385498046875], 
        'lp_bbox_score': 0.6572830677032471,
        'lp_number': None,
        'lp_text_score': None
        },
    4.0: {
        'car_bbox': [1956.56298828125, 1046.917724609375, 2509.57958984375, 1566.6651611328125], 
        'car_bbox_score': 0.8615961074829102,
        'lp_bbox': [222.3783721923828, 373.5968017578125, 370.99041748046875, 421.90069580078125], 
        'lp_bbox_score': 0.7343707084655762, 
        'lp_number': 'MU51TSU', 
        'lp_text_score': 0.035259282382269844
        }        
}

frame_value_flatten = {
    1 :{
        1.0: {

            'car_bbox': [678.2144775390625, 1232.2449951171875, 1315.8944091796875, 1791.5057373046875],
            'car_bbox_score': 0.894150972366333,
            'lp_bbox': [221.50367736816406, 379.7851257324219, 392.10955810546875, 429.76385498046875], 
            'lp_bbox_score': 0.6572830677032471,
            'lp_number': None,
            'lp_text_score': None
            },
        4.0: {
            'car_bbox': [1956.56298828125, 1046.917724609375, 2509.57958984375, 1566.6651611328125], 
            'car_bbox_score': 0.8615961074829102,
            'lp_bbox': [222.3783721923828, 373.5968017578125, 370.99041748046875, 421.90069580078125], 
            'lp_bbox_score': 0.7343707084655762, 
            'lp_number': 'MU51TSU', 
            'lp_text_score': 0.035259282382269844
            }        
    }
}
reframed_list = []
# df = pd.DataFrame(columns=['frame_number','track_id','car_bbox','car_bbox_score','lp_bbox','lp_bbox_score','lp_number','lp_text_score'])

for frame_number, tracks in frame_value_flatten.items():
    # print(frame_number)
    for track_id, data in tracks.items():
        # print(track_id)
        new_row = {'frame_number':frame_number,'track_id':track_id}
        new_row.update(data)
        reframed_list.append(new_row)
        # print(data)
        # print(new_row)
        # exit()
        # print(pd.DataFrame([new_row]))
        # exit()
        # df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
        # df.loc[-1] = new_row
        
        # df = df.append(pd.DataFrame([new_row]))
df = pd.DataFrame(reframed_list,columns=['frame_number','track_id','car_bbox','car_bbox_score','lp_bbox','lp_bbox_score','lp_number','lp_text_score'])
print(df)

exit()
# for i in value_flatten.keys():
#     print(i)

# exit()
# print(type(value))
# print(value_flatten)

# df = pd.DataFrame.from_dict(value,orient='index')
df = pd.DataFrame.from_dict(value_flatten,orient='index')
# df = pd.DataFrame(value_flatten)
print(df)
print(type(df.iloc[0]['car_bbox']))
# print(df)