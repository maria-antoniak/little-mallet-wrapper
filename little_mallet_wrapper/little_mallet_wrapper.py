from collections import defaultdict
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', font_scale=1.2)


STOPS = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
         'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
         'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
         'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
         'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
         'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
         'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
         'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
         'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
         'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
         'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
         'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 've', 'll', 'amp']



def print_dataset_stats(training_data):

    num_documents = len(training_data)
    mean_num_words = np.mean([len(d.split()) for d in training_data])
    vocab_size = len(list(set([w for d in training_data for w in d.split()])))

    print('Number of Documents:', num_documents)
    print('Mean Number of Words per Document:', round(mean_num_words, 1))
    print('Vocabulary Size:', vocab_size)


def process_string(text, 
                   lowercase=True, 
                   remove_short_words=True, 
                   remove_stop_words=True, 
                   remove_punctuation=True, 
                   numbers='replace', 
                   stop_words=STOPS,
                   stop_words_extra=[]):
    if lowercase:
        text = text.lower()
    if numbers == 'replace':
        text = re.sub('[0-9]+', 'NUM', text)
    elif numbers == 'remove':
        text = re.sub('[0-9]+', ' ', text)
    if remove_punctuation:
        text = re.sub(r'[^\sA-Za-z0-9À-ÖØ-öø-ÿЀ-ӿ/]', ' ', text)
    if remove_stop_words:
        text = ' '.join([word for word in text.split() if word not in stop_words + stop_words_extra])
    if remove_short_words:
        text = ' '.join([word for word in text.split() if not len(word) <= 2])
    text = ' '.join(text.split())
    return text


def quick_train_topic_model(path_to_mallet,
                            output_directory_path,
                            num_topics,
                            training_data):

    path_to_training_data           = output_directory_path + '/training.txt'
    path_to_formatted_training_data = output_directory_path + '/mallet.training'
    path_to_model                   = output_directory_path + '/mallet.model.' + str(num_topics)
    path_to_topic_keys              = output_directory_path + '/mallet.topic_keys.' + str(num_topics)
    path_to_topic_distributions     = output_directory_path + '/mallet.topic_distributions.' + str(num_topics)
    path_to_word_weights            = output_directory_path + '/mallet.word_weights.' + str(num_topics)

    import_data(path_to_mallet,
                path_to_training_data,
                path_to_formatted_training_data,
                training_data)                  
    train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      path_to_model,
                      path_to_topic_keys,
                      path_to_topic_distributions,
                      path_to_word_weights,
                      num_topics)
    
    topic_keys = load_topic_keys(path_to_topic_keys)
    topic_distributions = load_topic_distributions(path_to_topic_distributions)

    return topic_keys, topic_distributions


def import_data(path_to_mallet,
                path_to_training_data,
                path_to_formatted_training_data,
                training_data,
                training_ids=None,
                use_pipe_from=None):

    training_data_file = open(path_to_training_data, 'w')
    for i, d in enumerate(training_data):
        if training_ids:
            training_data_file.write(str(training_ids[i]) + ' no_label ' + d + '\n')
        else:
            training_data_file.write(str(i) + ' no_label ' + d + '\n')
    training_data_file.close()

    if use_pipe_from:
        print('Importing data using pipe...')
        os.system(path_to_mallet + ' import-file --input "' + path_to_training_data + '"' 
                                             + ' --output "' + path_to_formatted_training_data + '"' \
                                             + ' --keep-sequence' \
                                             + ' --use-pipe-from "' + use_pipe_from + '"'
                                             + ' --preserve-case')
        
    else:
        print('Importing data...')
        os.system(path_to_mallet + ' import-file --input "' + path_to_training_data + '"' 
                                             + ' --output "' + path_to_formatted_training_data + '"' \
                                             + ' --keep-sequence'
                                             + ' --preserve-case')

    print('Complete')


def train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      path_to_model,
                      path_to_topic_keys,
                      path_to_topic_distributions,
                      path_to_word_weights,
                      num_topics):

    print('Training topic model...')
    os.system(path_to_mallet + ' train-topics --input "' + path_to_formatted_training_data + '"' \
                                          + ' --num-topics ' + str(num_topics) \
                                          + ' --inferencer-filename "' + path_to_model + '"' \
                                          + ' --output-topic-keys "' + path_to_topic_keys + '"' \
                                          + ' --output-doc-topics "' + path_to_topic_distributions + '"' \
                                          + ' --topic-word-weights-file "' + path_to_word_weights + '"' \
                                          + ' --diagnostics-file' \
                                          + ' --optimize-interval 10')

    print('Complete')


def load_topic_keys(topic_keys_path):
    return [line.split('\t')[2].split() for line in open(topic_keys_path, 'r')]


def load_topic_distributions(topic_distributions_path):
    topic_distributions = []
    for line in open(topic_distributions_path, 'r'):
        if line.split()[0] != '#doc':
            instance_id, distribution = (line.split('\t')[1], line.split('\t')[2:])
            distribution = [float(p) for p in distribution]
            topic_distributions.append(distribution)
    return topic_distributions


def load_training_ids(topic_distributions_path):
    training_ids = []
    for line in open(topic_distributions_path, 'r'):
        if line.split()[0] != '#doc':
            instance_id, distribution = (line.split('\t')[1], line.split('\t')[2:])
            training_ids.append(instance_id)
    return training_ids


def load_topic_word_distributions(word_weight_path):
    
    topic_word_weight_dict = defaultdict(lambda: defaultdict(float))
    topic_sum_dict = defaultdict(float)
    with open(word_weight_path,'r') as f:       
        for _line in f:        
            _topic, _word, _weight = _line.split('\t')
            topic_word_weight_dict[_topic][_word] = float(_weight)
            topic_sum_dict[_topic] += float(_weight)

    topic_word_probability_dict = defaultdict(lambda: defaultdict(float))
    for _topic, _word_weight_dict in topic_word_weight_dict.items():
        for _word, _weight in _word_weight_dict.items():
            topic_word_probability_dict[_topic][_word] = _weight / topic_sum_dict[_topic]

    return topic_word_probability_dict


def get_top_docs(training_data, topic_distributions, topic_index, n=5):
    sorted_data = sorted([(_distribution[topic_index], _document) 
                          for _distribution, _document 
                          in zip(topic_distributions, training_data)], reverse=True)
    return sorted_data[:n]


def plot_categories_by_topics_heatmap(labels, 
                                      topic_distributions, 
                                      topic_keys, 
                                      output_path=None,
                                      target_labels=None,
                                      dim=None):
    
    # Combine the labels and distributions into a list of dictionaries.
    dicts_to_plot = []
    for _label, _distribution in zip(labels, topic_distributions):
        if not target_labels or _label in target_labels:
            for _topic_index, _probability in enumerate(_distribution):
                dicts_to_plot.append({'Probability': float(_probability),
                                      'Category': _label,
                                      'Topic': 'Topic ' + str(_topic_index).zfill(2) + ': ' + ' '.join(topic_keys[_topic_index][:5])})

    # Create a dataframe, format it for the heatmap function, and normalize the columns.
    df_to_plot = pd.DataFrame(dicts_to_plot)
    df_wide = df_to_plot.pivot_table(index='Category', 
                                     columns='Topic', 
                                     values='Probability')
    df_norm_col=(df_wide-df_wide.mean())/df_wide.std()
        
    # Show the final plot.
    if dim:
        plt.figure(figsize=dim)
    sns.set(style='ticks', font_scale=1.2)
    ax = sns.heatmap(df_norm_col, cmap=sns.cm.rocket_r)    
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xticks(rotation=30, ha='left')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()
    

def plot_categories_by_topic_boxplots(labels, 
                                      topic_distributions, 
                                      topic_keys, 
                                      target_topic_index,
                                      output_path=None,
                                      target_labels=None,
                                      dim=None):
    
    if not target_labels:
        target_labels = list(set(labels))
                   
    # Combine the labels and distributions into a dataframe.
    dicts_to_plot = []
    for _label, _distribution in zip(labels, topic_distributions):
        if not target_labels or _label in target_labels:
            dicts_to_plot.append({'Probability': float(_distribution[target_topic_index]),
                                  'Category': _label,
                                  'Topic': ' '.join(topic_keys[target_topic_index][:5])})
    df_to_plot = pd.DataFrame(dicts_to_plot)

    # Show the final plot.
    if dim:
        plt.figure(figsize=dim)
    sns.set(style='ticks', font_scale=1.2)
    sns.boxplot(data=df_to_plot,
                x='Category',
                y='Probability',
                color='skyblue')  
    sns.despine()
    plt.xticks(rotation=45, ha='right')
    plt.title('Topic: ' + ' '.join(topic_keys[target_topic_index][:5]))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def divide_training_data(documents, num_chunks=10):

    divided_documents = []
    document_ids = []
    times = []

    for doc_id, text in enumerate(documents):

        t = 0

        for _chunk in np.array_split(np.asarray(text.split()), num_chunks): 
            divided_documents.append(' '.join(_chunk))
            document_ids.append(doc_id)
            times.append(t)
            t += 0.1
        
    return divided_documents, document_ids, times


def infer_topics(path_to_mallet,
                 path_to_original_model,
                 path_to_new_formatted_training_data,
                 path_to_new_topic_distributions):

    print('Inferring topics using pre-trained model...')
    os.system(path_to_mallet + ' infer-topics --input "' + path_to_new_formatted_training_data + '"' \
                                          + ' --num-iterations 100' \
                                          + ' --inferencer "' + path_to_original_model + '"' \
                                          + ' --output-doc-topics "' + path_to_new_topic_distributions + '"')
    print('Complete')


def plot_topics_over_time(topic_distributions, topic_keys, times, topic_index, output_path=None):
    
    data_dicts = []
    for j, _distribution in enumerate(topic_distributions):        
        for _topic, _probability in enumerate(_distribution):
            if _topic == topic_index:
                data_dicts.append({'Probability': _probability,
                                   'Time': times[j]})
    data_df = pd.DataFrame(data_dicts)

    sns.set(style='ticks', font_scale=1.2)
    plt.figure(figsize=(7,2.5))
    sns.lineplot(data=data_df, 
                 x='Time', 
                 y='Probability', 
                 color='cornflowerblue') 
    plt.xlabel('Time')
    plt.ylabel('Topic Probability')
    plt.title(' '.join(topic_keys[topic_index][:5]))
    plt.tight_layout()
    sns.despine()
    if output_path:
        plt.savefig(output_path)
    plt.show()


def get_js_divergence_documents(document_index_1, document_index_2, topic_distributions):
    return jensenshannon(topic_distributions[document_index_1], topic_distributions[document_index_2])


def get_js_divergence_topics(topic_index_1, topic_index_2, topic_word_probability_dict):
    vocab = list(set(list(topic_word_probability_dict[topic_index_1].keys()) + list(topic_word_probability_dict[topic_index_2].keys())))
    dist1 = [topic_word_probability_dict[topic_index_1][w] for w in vocab]
    dist2 = [topic_word_probability_dict[topic_index_2][w] for w in vocab]
    return jensenshannon(dist1, dist2)