result = {}
result[0] = {}

result[0][1] = {
                            'car': {
                                'bbox': [1, 1, 2, 22],
                                'bbox_score': 10
                            },
                            'license_plate': {
                                'bbox': [],
                                'bbox_score': "plate_score",
                                'number': "lic_text",
                                'text_score': "lic_score"
                            }
                } 

result[0][4] = {
                            'car': {
                                'bbox': [1, 1, 2, 22],
                                'bbox_score': 10
                            },
                            'license_plate': {
                                'bbox': [],
                                'bbox_score': "plate_score",
                                'number': "lic_text",
                                'text_score': "lic_score"
                            }
                } 

print(result[0].keys())

for i in result[0].keys():
    print(i)
    print(result[0][i]["license_plate"]["number"])