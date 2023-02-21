'''
log 教學: https://stylengineer.com/program/python-logging-guide/#%e4%bb%80%e9%ba%bc%e6%98%aflog-level
'''
import os
import logging

def getMyLogger(logFileName, logDir, fileRecordLevel):
    '''
    logFileName: log檔案的名字
    logDirL 存log的資料夾
    fileRecordLevel: log檔案裡面要記錄哪種等級的資訊
    預設stream上面都是顯示 INFO等級
    '''

    RecordLevel = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }

    #1.setup log path and create log directory
    logName = logFileName
    logDir = logDir
    logPath = logDir + '/' + logName
    #create log directory 
    os.makedirs(logDir,exist_ok=True)

    #2.create logger, then setLevel
    allLogger = logging.getLogger(logName) #allLogger是自己取的name
    allLogger.setLevel(logging.DEBUG)

    #3.create file handler, then setLevel
    #create file handler

    fileHandler = logging.FileHandler(logPath,mode='w') #mode a代表append, w代表每次砍舊重寫
    fileHandler.setLevel(RecordLevel[fileRecordLevel])

    #4.create stram handler, then setLevel
    #create stream handler
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(logging.INFO)

    #5.create formatter, then handler setFormatter
    AllFormatter = logging.Formatter("%(asctime)s - [line:%(lineno)d] - %(levelname)s: %(message)s")
    fileHandler.setFormatter(AllFormatter)
    streamHandler.setFormatter(AllFormatter)

    #6.logger addHandler
    allLogger.addHandler(fileHandler)
    allLogger.addHandler(streamHandler)

    return allLogger