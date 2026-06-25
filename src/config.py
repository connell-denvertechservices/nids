"""Central configuration: paths, schema, and constants.

No logic lives here — only constants the rest of the package imports.
Keeping the brittle NSL-KDD schema in one place keeps other modules clean.
"""
from pathlib import Path

# Paths
# config.py lives in src/, so the project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"


TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

#Public mirror of the NSL-KDD dataset raw text files, in case the original source is unavailable
TRAIN_URL = (
    "https://raw.githubusercontent.com"
    "/jmnwong/NSL-KDD-Dataset/master/KDDTrain+.txt"
)
TEST_URL = (
    "https://raw.githubusercontent.com"
    "/jmnwong/NSL-KDD-Dataset/master/KDDTest+.txt"
)
# Schema
# The 41 features in the fixed order they appear in every row, followed by the label and difficulty score (source: NSL-KDD dataset documentation)
COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

CATEGORICAL_COLUMNS = [
    "protocol_type",
    "service",
    "flag"]
DROP_COLUMNS = ["difficulty"] # 'label' is the target variable, so we keep it, but 'difficulty' is not relevant to our modeling task

#Attack Taxonomy
#NSL-KDD labels are specific attack names. This maps to one of four 
#attack catagories (used for the optional 5-class task). Antyhing labeld
#'normal' stays normal. Names not in this map default to 'unknown' (though there are none in the dataset)
ATTACK_CATEGORY = {
    # DoS
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos",
    "apache2": "dos", "processtable": "dos", "udpstorm": "dos",
    "worm": "dos",
    #Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # R2L (remote to local)
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l",
    "warezclient": "r2l", "warezmaster": "r2l", "sendmail": "r2l",
    "named": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R (user to root)
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

#Modeling
RANDOM_STATE = 42
TEST_SIZE = 0.2 # used only if we split the train file for validation

