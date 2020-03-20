import os
import re

import numpy as np
import pandas as pd

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


def remove_extra_spaces(text):
    return ' '.join(text.split())


def remove_non_alpha(text):
    text = re.sub('[0-9]+', 'NUM', text)
    return re.sub('[^A-Za-z\s]', ' ', text)


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word not in STOPS])


def remove_short_words(text, min_length):
    return ' '.join([word for word in text.split() if not len(word) < min_length])


def process_string(text):
    text = text.lower()
    text = remove_non_alpha(text)
    text = remove_stop_words(text)
    text = remove_short_words(text, 2)
    text = remove_extra_spaces(text)
    return text


def train_topic_model(mallet_path,
                      training_data_path,
                      formatted_training_data_path,
                      model_path,
                      topic_keys_path,
                      topic_distributions_path,
                      training_data,
                      num_topics):

    training_data_file = open(training_data_path, 'w')
    for i, d in enumerate(training_data):
        training_data_file.write(str(i) + ' ' + str(i) + ' ' + d + '\n')

    print('Importing data...')
    os.system(mallet_path + ' import-file --input "' + training_data_path + '"' 
                                      + ' --output "' + formatted_training_data_path + '"' \
                                      + ' --keep-sequence')

    print('Training topic model...')
    os.system(mallet_path + ' train-topics --input "' + formatted_training_data_path + '"' \
                                       + ' --num-topics ' + str(num_topics) \
                                       + ' --inferencer-filename "' + model_path + '"' \
                                       + ' --output-topic-keys "' + topic_keys_path + '"' \
                                       + ' --output-doc-topics "' + topic_distributions_path + '"')


def load_topic_keys(topic_keys_path):
    return [line.split('\t')[2].split() for line in open(topic_keys_path, 'r')]


def load_topic_distributions(topic_distributions_path):
    topic_distributions = []
    for line in open(topic_distributions_path, 'r'):
        distribution = line.split('\t')[2:]
        distribution = [float(p) for p in distribution]
        topic_distributions.append(distribution)
    return topic_distributions


# def get_top_docs(training_data, topic_distributions, topic, n=5):




def plot_categories_by_topics_heatmap(labels, 
                                      distributions, 
                                      topics, 
                                      output_path=None,
                                      target_labels=None,
                                      dim=None):
    
    # Combine the labels and distributions into a list of dictionaries.
    dicts_to_plot = []
    for _label, _distribution in zip(labels, distributions):
        if not target_labels or _label in target_labels:
            for _topic_index, _probability in enumerate(_distribution):
                dicts_to_plot.append({'Probability': float(_probability),
                                      'Category': _label,
                                      'Topic': ' '.join(topics[_topic_index][:5])})

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
    sns.heatmap(df_norm_col, cmap=sns.cm.rocket_r)    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    plt.show()
    

def plot_categories_by_topic_boxplots(labels, 
                                      distributions, 
                                      topics, 
                                      target_topic_index,
                                      output_path=None,
                                      target_labels=None,
                                      dim=None):
    
    if not target_labels:
        target_labels = list(set(labels))
                   
    # Combine the labels and distributions into a dataframe.
    dicts_to_plot = []
    for _label, _distribution in zip(labels, distributions):
        if not target_labels or _label in target_labels:
            dicts_to_plot.append({'Probability': float(_distribution[target_topic_index]),
                                  'Category': _label,
                                  'Topic': ' '.join(topics[target_topic_index][:5])})
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
    plt.title('Topic: ' + ' '.join(topics[target_topic_index][:5]))
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
            t += 10
        
    return divided_documents, document_ids, times


def infer_topics(mallet_path,
                 original_formatted_training_data_path,
                 original_model_path,
                 new_training_data_path,
                 new_formatted_training_data_path,
                 new_topic_distributions_path,
                 new_training_data):

    new_training_data_file = open(new_training_data_path, 'w')
    for i, d in enumerate(new_training_data):
        new_training_data_file.write(str(i) + ' ' + str(i) + ' ' + d + '\n')

    print('Importing data...')
    os.system(mallet_path + ' import-file --input "' + new_training_data_path + '"'
                                      + ' --output "' + new_formatted_training_data_path + '"' \
                                      + ' --keep-sequence' \
                                      + ' --use-pipe-from "' + original_formatted_training_data_path + '"')

    print('Inferring topics using pre-trained model...')
    os.system(mallet_path + ' infer-topics --input "' + new_formatted_training_data_path + '"' \
                                       + ' --num-iterations 100' \
                                       + ' --inferencer "' + original_model_path + '"' \
                                       + ' --output-doc-topics "' + new_topic_distributions_path + '"')


def plot_topics_over_time(topic_distributions, topics, times, topic_index, output_path=None):
    
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
    plt.title(' '.join(topics[topic_index][:5]))
    plt.tight_layout()
    sns.despine()
    if output_path:
        plt.savefig(output_path)
    plt.show()