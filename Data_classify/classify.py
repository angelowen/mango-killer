import csv,os
import shutil
path = "./C1-P1_Train/"
try:
  os.mkdir(path+"A")
  os.mkdir(path+"B")
  os.mkdir(path+"C")
except:
  pass
with open('train.csv', newline='') as csvfile:
  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)
  # 以迴圈輸出每一列
  for row in rows:
    try:
      shutil.move(path+row[0], path+row[1])
    except:
      pass
   