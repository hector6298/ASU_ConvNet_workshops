from itertools import compress
def clean_annotation(annotations, train_dir, test_dir):
  files = []
  for r, d, f in os.walk(train_dir):
    for file in f:
      #print(os.path.join(r, file))
      files.append(os.path.join(r, file))
  for r, d, f in os.walk(test_dir):
    for file in f:
      #print(os.path.join(r, file))
      files.append(os.path.join(r, file))

  mask = np.ones(len(annotations), dtype=bool)

  for i in range(len(annotations)):
    if f"{annotations[i][1]}" not in files:
      print(f"Deleting {annotations[i][1]}")
      mask[i] = False
  return list(compress(annotations, mask)), files

def _process_csv_file(file):
    with open(file, 'r') as fr:
        files = fr.readlines()
    return files

def generate_train_test_filenames():

    print("Generating Train-Test split")
    TRAINFILE = '/content/train_split_v3.txt'
    ESTFILE = '/content/test_split_v3.txt'
    dataset_train = _process_csv_file(TRAINFILE)
    dataset_test = _process_csv_file(TESTFILE)
    datasets = {'normal': [], 'pneumonia': [], 'COVID-19': []}
    for l in dataset_train:
    entry = l.split()
    entry[1] = f"data/train/{entry[1]}"
    datasets[entry[2]].append(entry)
    for l in dataset_test:
    entry = l.split()
    entry[1] = f"data/test/{entry[1]}"
    datasets[entry[2]].append(entry)

    break_point_normal = int(len(datasets['normal'])/5)
    break_point_covid = int(len(datasets['COVID-19'])/5)
    break_point_pneumonia = int(len(datasets['pneumonia'])/5)

    train_set = [datasets['normal'][0:(break_point_normal*4)] + datasets['pneumonia'][0:(break_point_pneumonia*4)] + datasets['COVID-19'][0:(break_point_covid*4)]]
    test_set = [datasets['normal'][(break_point_normal*4):] + datasets['pneumonia'][(break_point_pneumonia*4):] + datasets['COVID-19'][(break_point_covid*4):]]

    print("Cleaning train annotations...")
    annotations_train, filestr = clean_annotation(train_set[0], "data/train", "data/test")
    print("Cleaning test annotations...")
    annotations_test, fileste = clean_annotation(test_set[0], "data/train", "data/test")
    print('done!')