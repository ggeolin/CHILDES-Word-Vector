import nltk
import pickle
import json
import glob
import gensim

from os import replace, system
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.util import pr
from tqdm import tqdm

STOPWORDS = set(stopwords.words('english'))
NUMS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']

# print a line of stars
def printLine():
    for i in range(50):
        print('*', end = '')
    print('\n')

# clear out the screen
def clear():
    system('clear')

# load the config.txt
def load_config():
    system('clear')

# method to generate the word2vec model
def make_model():
    clear()

    # error management
    try:
        with open('config.json', 'r') as config:
            config_file = json.load(config)
            print(('\033[4m' + 'STATUS' + '\033[0m').center(58, '*'))
            print('\033[92m' + 'OK: config file is successfully loaded' + '\033[0m\n')
    except:
        print('\033[91m' + 'ERROR: Cannot load file' + '\033[0m')
        return NextState()

    printLine()

    all_files = glob.glob(config_file['input_dir'] + '/*.cha')
    if len(all_files) == 0:
        print('\033[91m' + 'ERROR: Input Directory is Empty' + '\033[0m')
        return NextState()
    
    content = ''

    # load all the file into content
    for file_i in tqdm(all_files, bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        with open(file_i, 'r') as fd:
            for line in fd:
                if line.startswith('*CHI:\t'):
                    num_flag = False
                    # if the line has number, continue
                    for number in NUMS:
                        if number in line:
                            num_flag = True
                    
                    if not num_flag:
                        new_line = line.replace('*CHI:\t', '').replace('_', ' ').replace('+', ' ')
                        if '.' in line:
                            content += (new_line.split('.'))[0] + '.\n'
                        elif '?' in line:
                            content += (new_line.split('?'))[0] + '.\n'

    print('\033[92m' + 'OK: All files are successfully loaded' + '\033[0m\n')

    # remove all stopwords in the content set
    for stop_word in tqdm(STOPWORDS, bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        if stop_word in content:
            content.replace(stop_word, '')
    print('\033[92m' + 'OK: All stopwords removed' + '\033[0m\n')

    sentences = nltk.sent_tokenize(content.lower())
    data = []
    
    for sent in tqdm(sentences, bar_format = '{l_bar}{bar:30}{r_bar}{bar:-10b}'):
        data.append(nltk.word_tokenize(sent))
    print('\033[92m' + 'OK: All words are tokenized' + '\033[0m\n')

    printLine()

    # generate the model
    model_file = gensim.models.Word2Vec(data, 
    min_count = config_file['min_count'], 
    size = config_file['size'],
    window = config_file['window'])

   

    print(model_file)
    pickle.dump(model_file, open(config_file['model_name'], 'wb'))
    print('\033[92m' + 'OK: Model file is successfully generated' + '\033[0m')

    return NextState()

def diff_words():
    clear()

    # error management
    print(('\033[4m' + 'STATUS' + '\033[0m').center(54, '*'))
    try:
        with open('config.json', 'r') as config:
            config_file = json.load(config)
            print('\033[92m' + 'OK: config file is successfully loaded' + '\033[0m')

        with open(config_file['model_name'], 'rb') as model_file:
            model = pickle.load(model_file)
            print('\033[92m' + 'OK: Model file is successfully loaded' + '\033[0m\n')
    except:
        print('\033[91m' + 'ERROR: Cannot load file(s)' + '\033[0m')
        return NextState()

    print(('\033[4m' + 'Find Two Words Similarities' + '\033[0m').center(54, '*'))
    print('*** Enter 2 words' +  '\033[94m' + ' OR ' + '\033[0m' + 'Press [ENTER] to exit ***')

    while True:
        word_1 = input('Word 1: ')
        if word_1 == '':
            break

        word_2 = input('Word 2: ')
        if word_2 == '':
            break

        try:
            print('[', word_1, ',' , word_2, ']:', model.wv.similarity(word_1, word_2) , '\n')
        except KeyError as err:
            print('\033[91m' + 'ERROR:' + '\033[0m' , err , '\n') 

    return NextState()

def find_words():
    clear()

    # error management
    print( ('\033[4m' + 'STATUS' + '\033[0m').center(51, '*') )
    try:
        with open('config.json', 'r') as config:
            config_file = json.load(config)
            print('\033[92m' + 'OK: config file is successfully loaded' + '\033[0m')

        with open(config_file['model_name'], 'rb') as model_file:
            model = pickle.load(model_file)
            print('\033[92m' + 'OK: Model file is successfully loaded' + '\033[0m\n')
    except:
        print('\033[91m' + 'ERROR: Cannot load file(s)' + '\033[0m')
        return NextState()

    print( ('\033[4m' + 'Find All Similar Words' + '\033[0m').center(51,'*') )
    print('*** Enter word' +  '\033[94m' + ' OR ' + '\033[0m' + 'Press [ENTER] to exit ***')

    while True:
        word = input('Word: ')
        if word == '':
            break

        try:
            similar_words = model.wv.most_similar(positive=[word])
            print('[', word, ']:')
            for i in similar_words:
                line = []
                line.append(str(i[0]))
                line.append(str(i[1]))
                print('{:<12} : {:<20}'.format(*line))
            print()

        except KeyError as err:
            print('\033[91m' + 'ERROR:' + '\033[0m' , err , '\n') 


    return NextState()

def output_vocab():
    clear()

    # error management
    try:
        with open('config.json', 'r') as config:
            config_file = json.load(config)
            print('\033[92m' + 'OK: config file is successfully loaded' + '\033[0m')

        with open(config_file['model_name'], 'rb') as model_file:
            model = pickle.load(model_file)
            print('\033[92m' + 'OK: Model file is successfully loaded' + '\033[0m\n')
    except:
        print('\033[91m' + 'ERROR: Cannot load file(s)' + '\033[0m')
        return NextState()

    vocab = list(model.wv.vocab)
    for i in vocab:
        print(i)

    return NextState()

def output_n_words():
    clear()

    # error management
    print( ('\033[4m' + 'STATUS' + '\033[0m').center(48, '*') )
    try:
        with open('config.json', 'r') as config:
            config_file = json.load(config)
            print('\033[92m' + 'OK: config file is successfully loaded' + '\033[0m')

        with open(config_file['model_name'], 'rb') as model_file:
            model = pickle.load(model_file)
            print('\033[92m' + 'OK: Model file is successfully loaded' + '\033[0m\n')
    except:
        print('\033[91m' + 'ERROR: Cannot load file(s)' + '\033[0m')
        return NextState()

    print( ('\033[4m' + 'Output n frequent words' + '\033[0m').center(48,'*') )
    print('*** Enter n' +  '\033[94m' + ' OR ' + '\033[0m' + 'Press [ENTER] to exit ***\n')

    n = -1
    while True:
        user_in = input('Enter top n words to output: ')
        if user_in.isdigit():
            n = int(user_in)
            break
        elif user_in == '':
            return NextState()
    
    word_counts = {}
    for word in model.wv.vocab:
        word_counts[word] = model.wv.vocab[word].count

    num_output = 0
    for word in sorted(word_counts, key=word_counts.get, reverse=True):
        if num_output == n:
            break
        print(word, word_counts[word])
        num_output += 1

    return NextState()

def NextState():
    prompt = input('\nEnter [1] Main Menu\t' + '[ENTER] Exit\n$ ')

    if prompt == '1':
        return True
    else:
        return False

def print_options():
    clear()

    print('\033[4m' + 'CHILDES MODEL' + '\033[0m')

    # print options
    print(
        'OPTIONS:\n',
        '[1]: Generate Model\n',
        '[2]: Compare words\n',
        '[3]: Find related words\n',
        '[4]: Output all words\n',
        '[5]: Output top n words'
    )

    print('** Press [ENTER] To Exit **')

def main():
    next_state = True
    flag = True

    while flag and next_state:
        print_options()
        select = input('\033[92m' + 'Pick An Option: ' + '\033[0m')
        
        if select == '1':
            next_state = make_model()
        elif select == '2':
            next_state = diff_words()
        elif select == '3':
            next_state = find_words()
        elif select == '4':
            next_state = output_vocab()
        elif select == '5':
            next_state = output_n_words()
        else:
            flag = False


    print('\033[94m' + 'Bye...' + '\033[0m')


main()
