import sys
from pathlib import Path
from shutil import copytree, ignore_patterns, copyfile




# This script initializes new pytorch project with the template files.
# Run `python3 new_project.py ../MyNewProject` then new project named 
# MyNewProject will be made
current_dir = Path()
assert (current_dir / 'new_project.py').is_file(), 'Script should be executed in the pytorch-template directory'
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject'

project_name = Path(sys.argv[1])
target_dir = current_dir.parent / project_name

ignore = [".git", "data", "saved", "LICENSE", ".flake8", "README.md", "__pycache__"]
should_copy = ["base", "models", "trainer", "train.py", "utils", "new_project.py"]
for item in should_copy:
    assert (current_dir / item).exists(), f'{item} does not exist in the template directory'
    if (current_dir / item).is_dir():
        copytree(current_dir / item, target_dir / item, ignore=ignore_patterns(*ignore), dirs_exist_ok=True)
    else:
        copyfile(current_dir / item, target_dir / item)

print('New project initialized at', target_dir.absolute().resolve())