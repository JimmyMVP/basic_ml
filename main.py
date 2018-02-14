from map_spam import MAPSpamDetector
import pandas as pd
import numpy as np

df = pd.read_csv('./spam.csv', encoding='latin-1')

df['v1'] = df['v1'].apply(lambda x: 0 if x == 'ham' else 1)

df_test = df.sample(1000)
df_train = df
df_train.drop(df_test.index)

train_smss = df_train.v2
train_labels = df_train.v1
test_smss = df_test.v2
test_labels = df_test.v1

detector = MAPSpamDetector()
detector.train(train_smss, train_labels)

num_correct = 0
for sms, l in zip(test_smss, test_labels):
    pred = np.argmax(detector.infer(sms))
    num_correct+= 1 if pred == l else 0

print("Error rate test set: ", 1. - num_correct/len(test_smss))

