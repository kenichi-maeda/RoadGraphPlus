import os
import json

# Fixes the data files, so they work on anyone's machine
def fix_image_paths(old_name='kmaeda2', new_name='dbchanin'):
    labels_dir = 'data/labels'

    i = 0
    for filename in os.listdir(labels_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(labels_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Replace kmaeda2 with new_name in image_path
            if 'image_path' in data:
                data['image_path'] = data['image_path'].replace('kmaeda2', new_name)

            # Write back the modified json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)

            i += 1
            if i % 100 == 0:
                print(f'Processed {i} files.')
                print(f'Currently on file: {filename}')

    # Remove old_username within shell scripts
    script_files = ['trainerH.sh', 'eval.sh', 'trainer.sh','eval.py','trainer.py','trainerH.py']
    for script in script_files:
        if os.path.exists(script):
            with open(script, 'r') as f:
                script_data = f.read()

            script_data = script_data.replace(old_name, new_name)

            with open(script, 'w') as f:
                f.write(script_data)

            print(f'Updated {script} to replace {old_name} with {new_name}.')

def fix_conda_environments(old_conda='csci2952_mocov3', new_conda='csci2470_final_project'):
    ssh_files = ['trainerH.sh', 'eval.sh', 'trainer.sh']
    for ssh_file in ssh_files:
        if os.path.exists(ssh_file):
            with open(ssh_file, 'r') as f:
                ssh_data = f.read()

            ssh_data = ssh_data.replace(old_conda, new_conda)

            with open(ssh_file, 'w') as f:
                f.write(ssh_data)

            print(f'Updated {ssh_file} to replace {old_conda} with {new_conda}.')

if __name__ == '__main__':
    OLD_USERNAME, NEW_USERNAME = 'kmaeda2', 'dbchanin'
    fix_image_paths(old_name=OLD_USERNAME, new_name=NEW_USERNAME)

    OLD_CONDA, NEW_CONDA = 'csci2952_mocov3', 'csci2470_final_project'
    fix_conda_environments(old_conda=OLD_CONDA, new_conda=NEW_CONDA)