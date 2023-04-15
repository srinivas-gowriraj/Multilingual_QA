from datasets import load_dataset
val_eng = []
val_eng.append(load_dataset('json', data_files="/usr0/home/sohamdit/Multilingual_QA/src/data/splits/EnDoc2BotRetrieval_train.json")["train"])
val_eng = [y for dataset in val_eng for y in dataset]

val_eng = [i for i in  val_eng if i["positive"].split("//")[-1]== ("en-DepartmentOfMotorVehicles")]

print(len(val_eng))