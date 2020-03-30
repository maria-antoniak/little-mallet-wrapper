# little-mallet-wrapper

This is a little Python wrapper around the topic modeling functions of [MALLET](http://mallet.cs.umass.edu/topics.php).

Currently under construction; please send feedback/requests to Maria Antoniak.

See demo.ipynb for a demonstration of how to use the functions in little-mallet-wrapper.


## Documentation

### print_dataset_stats(training_data)

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| training_data      | list of strings   | Documents that will be used to train the topic model. |

### process_string(text, lowercase=True, remove_short_words=True, remove_stop_words=True, remove_punctuation=True, numbers='replace', stop_words=STOPS)

| Name               | Type              | Description                      |
| ------------------ | ----------------- | -------------------------------- |
| text      | string   | Individual document to process. |
| lowercase | boolean  | Whether or not to lowercase the text. |
| remove_short_words | boolean | Whether or not to remove words with fewer than 2 characters. |
| remove_stop_words | boolean | Whether or not to remove stopwords. |
| remove_punctuation | boolean | Whehter or not to remove punctuation (not A-Za-z0-9) |
| remove_numbers | string | 'replace' replaces all numbers with the normalized token NUM; 'remove' removes all numbers. |
| stop_words | list of strings | Custom list of words to remove. |
| RETURNS | string | Processed version of the input text. |

### import_data

### train_topic_model

### load_topic_keys

### load_topic_distributions

### get_top_docs

### plot_categories_by_topics_heatmap

### plot_categories_by_topic_boxplots

### divide_training_data

### infer_topics

### plot_topics_over_time
