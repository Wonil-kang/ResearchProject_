import os
from pathlib import Path

logPath = "log"
logPathResult = "log/result.log"

def logResult(text):

    checkDirectory()

    f = open(logPathResult, 'a')

    f.write(text + "\n")
    f.close()

def checkDirectory():
    if not os.path.exists(logPath):
        os.mkdir("log")
