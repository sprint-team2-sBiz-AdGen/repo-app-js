DB_HOST = "postgres"
DB_PORT = "5432"
DB_NAME = "feedlyai"
DB_USER = "feedlyai"
DB_PASSWORD = "feedlyai_dev_password_74154"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

import os
ASSETS_DIR = "/opt/feedlyai/assets"
PART_NAME = "js"