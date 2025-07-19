import os

DISCOVERSE_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if os.getenv('DISCOVERSE_ASSETS_DIR'):
    DISCOVERSE_ASSETS_DIR = os.getenv('DISCOVERSE_ASSETS_DIR')
    print(f'>>> get env "DISCOVERSE_ASSETS_DIR": {DISCOVERSE_ASSETS_DIR}')
else:
    DISCOVERSE_ASSETS_DIR = os.path.join(DISCOVERSE_ROOT_DIR, 'models')

__version__ = "1.8.5"
__logo__ = """
    ____  _________ __________ _    ____________  _____ ______
   / __ \/  _/ ___// ____/ __ \ |  / / ____/ __ \/ ___// ____/
  / / / // / \__ \/ /   / / / / | / / __/ / /_/ /\__ \/ __/   
 / /_/ // / ___/ / /___/ /_/ /| |/ / /___/ __ _/___/ / /___   
/_____/___//____/\____/\____/ |___/_____/_/ |_|/____/_____/   
"""