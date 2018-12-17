# Named Entity Recognition in medical prescription text using BiLSTM and charecter embeddings

## Data Preprocessing
As for any natural language processing
task, we need a word embedding. created
one from the text data rather than using Glove since this
dataset has unique medical text data. Keras Tokenizer was used
create a embedding by fitting on the sentences from the input text
files.

The Events are categorized in to 7 types and each event can be a
single word or multiple words. A dataset created from extent file has
been merged with an empty dataset from text files on start and end
position. Since there are single word and multi word events, BIO
tagging has been used on the classes so maintain the relationships
between events and help with prediction accuracy. Used
character embedding along with word embeddings to add context
on a character level because of the way medical terms are named.
Padding the character array and the word arrays since
every word and sentence is of different length and we need them to
be of uniform length to be able to use it as input to our network. Followed similar procedure for the test data, every text input file
has been processed to a dataframe and used for prediction.

## Architecture

![Alt text](res/model.png?raw=true "Architecture")

The character inputs given are padded
sequences of characters of length 20. The work inputs are a padded
sequence of size tokens with max sentence length set to 100. The
character embedding layer takes the input and generates a charac-
ter embedding of dimension of 500. The output of this character
embedding is fed to a BiLSTM of size 50.We decided to use BiLSTM
network for the characters due to the importance of the character
sequences in the medical text.The output of character BiLSTM is
concatenated with the word embeddings and passed as input to the
second BiLSTM layer of size 50. Final fully connected layer with
softmax layer is used to get the desired number of classes.The net-
work is trained for 30 epochs and saved to a hdf5 file to be reused
for testing. There is also commented code for using cnn on charecters rather than using BiLSTM, from the experiments performed on this dataset BiLSTM seems to work best


## Results
The evaluation has been done by the evaluation script on the test
data given along with the challenge. The evaluation script has an
option to just evaluate event tags and used that to get
the outputs below. For getting the best possible performance we
have worked on multiple architecture for the model and decided
on BILSTM char and BiLSTM word architecture. the table below
shows accuracies obtained by 3 different architectures.
Besides just trying different architectures we have tried tuning
the hyper parameters for each of them and choose the most optimal values for each.


| Architecture | Precision | Recall | F1 SCore |
| :---: | :---: | :---: | :---: |
| Word BiLSTM | 0.7931 | 0.8237 | 0.8081 |
| Word BiLSTM + Char CNN | 0.7753 | 0.8681 | 0.8191 |
| Word BiLSTM + Char BiLSTM | 0.811 | 0.8546 | 0.8322 |



Example, the number of epochs our final model has
been decided by monitoring the performance of with increments
of 5 epochs. We have considered early stopping at 25 epochs since
there is no difference in F1 score of the model. different sized of
BiLSTM and word embedding have been tried before deciding the
values for our final model

| Epochs | Precision | Recall | F1 SCore |
| :---: | :---: | :---: | :---: |
| 15 | 0.7643 | 0.9037 | 0.8081 |
|20| 0.8035| 0.8546| 0.8283|
|25 |0.811 |0.8546| 0.8322|
|30 |0.8083| 0.8647| 0.8325|

The size of the sentences we consider also played an important role in giving better accuracy, although increasing the size is computationally intensive, the accuracy increase justifies it.

| length | Precision | Recall | F1 SCore |
| :---: | :---: | :---: | :---: |
| 50 | 0.715 | 0.8557 | 0.779 |
| 75 | 0.811 |0.8546 |0.8322 |
| 100 | 0.7873 | 0.8857 | 0.8336 |

## Areas of improvement

Although this has achieved good performance with the BiLSTM architecture adding some rule based in to this would help the performance a lot. Most of the output observed felt that it requires some rules from a subject matter expert in medical domain. This is mostly due to the fact that medical data is unique and different from normal conversational or any other text data. Although have tried multiple architectures and various combinations of hyper parameters, just due to sheer number of combinations possible, there is a good chance we might have missed something. Hyperparameter tuning could help with accuracy.
