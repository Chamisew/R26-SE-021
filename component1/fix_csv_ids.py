import pandas as pd

df = pd.read_csv('output/ml_ready_dataset.csv')
proj_map = {}
next_id = 1

def get_id(name):
    global next_id
    if name not in proj_map:
        proj_map[name] = f'project_{next_id}'
        next_id += 1
    return proj_map[name]

df['project_id'] = df['service_name'].apply(get_id)
df.to_csv('output/ml_ready_dataset.csv', index=False)
print('Fixed dataset with sequential IDs.')
