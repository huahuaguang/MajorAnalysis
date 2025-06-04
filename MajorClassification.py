import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_files
import random

random.seed(42)

import re


def clean_text(text):
    if isinstance(text, str):
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove special characters
        # w: word, including letters (a-z, A-Z), numbers (0-9), and underscores (_).
        # s: whitespace characters, including spaces, tabs (t), line breaks (n)
        text = re.sub(r'[^\w\s]', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove code blocks (assuming code blocks are indicated by indentation)
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove indentation from each line
        text = re.sub(r'\n\s*\n', '\n', text)  # Merge excess newline characters

        # Remove code snippets (assuming code snippets are indicated by specific syntax, such as ` or ``` )
        text = re.sub(r'`[^`]*`', '', text)  # Remove single-line code snippets
        text = re.sub(r'```[\s\S]*?```', '', text)  # Remove multi-line code snippets

        # Remove excess spaces and newline characters
        text = ' '.join(text.split())

        return text
    return ''


def process_20news(data_path, computer_related_groups):
    news_data = load_files(data_path, encoding='latin1', load_content=True)

    news_df = pd.DataFrame({
        'text': news_data['data'],
        'original_label': np.take(news_data['target_names'], news_data['target']),  # Fixed
        'source': '20news'
    })

    news_df['text'] = news_df['text'].str.replace('\n', ' ').str.replace('\s+', ' ', regex=True)
    news_df['is_computer_related'] = news_df['original_label'].apply(
        lambda x: 1 if any(group in x for group in computer_related_groups) else 0
    )
    return news_df


# 2. Process the StackOverflow dataset
def process_stackoverflow(train_path, valid_path):
    """
    Process the StackOverflow dataset, extracting titles, bodies, and labels
    """
    # Load the data
    so_train = pd.read_csv(train_path)
    so_valid = pd.read_csv(valid_path)

    # Merge the training and validation sets
    so_df = pd.concat([so_train, so_valid])

    # Select the required columns and rename them
    so_df = so_df[['Title', 'Body', 'Y']].copy()
    so_df.columns = ['title', 'text', 'original_label']

    # Clean the text - combine titles and bodies
    so_df['text'] = so_df['title'] + ' ' + so_df['text']
    so_df['text'] = so_df['text'].str.replace('\n', ' ').str.replace('\s+', ' ', regex=True)

    # Add source and computer-related labels (StackOverflow is assumed to be computer-related by default)
    so_df['source'] = 'stackoverflow'
    so_df['is_computer_related'] = 1

    return so_df


# 3. Main processing function
def create_final_dataset():
    # Define computer-related newsgroups
    computer_related_groups = [
        'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
        'comp.windows.x', 'comp.ai', 'comp.lang'
    ]

    # Process the 20news data
    news_train = process_20news('./datasets/20news-bydate/20news-bydate-train', computer_related_groups)
    news_test = process_20news('./datasets/20news-bydate/20news-bydate-test', computer_related_groups)
    news_full = pd.concat([news_train, news_test])

    # Sample from 20news - 3000 non-computer-related samples
    non_computer_news = news_full[news_full['is_computer_related'] == 0].sample(3000, random_state=42)
    news_sample = pd.concat([non_computer_news])

    # Process the StackOverflow data
    so_data = process_stackoverflow(
        './datasets/StackOverflow-QQDataset/train.csv',
        './datasets/StackOverflow-QQDataset/valid.csv'
    )

    # Sample 3000 entries from StackOverflow (sorted by quality, assuming there is a Score column)
    # If there is no Score column, sample randomly
    if 'Score' in so_data.columns:
        so_sample = so_data.sort_values('Score', ascending=False).head(3000)
    else:
        so_sample = so_data.sample(3000, random_state=42)

    # Merge the datasets
    final_df = pd.concat([
        news_sample[['text', 'original_label', 'source', 'is_computer_related']],
        so_sample[['text', 'original_label', 'source', 'is_computer_related']]
    ])

    # Reset the index
    final_df.reset_index(drop=True, inplace=True)

    # Apply the cleaning function
    final_df['text'] = final_df['text'].apply(clean_text)

    # Remove empty texts
    final_df = final_df[final_df['text'].str.len() > 0]

    # Save to CSV
    final_df.to_csv('./datasets/computer_related_text_dataset.csv', index=False)
    print("The dataset has been saved as computer_related_text_dataset.csv")
    print(f"Total number of samples: {len(final_df)}")
    print(f"Number of computer-related samples: {final_df['is_computer_related'].sum()}")
    print(f"Number of non-computer-related samples: {len(final_df) - final_df['is_computer_related'].sum()}")



# Execute the processing
if __name__ == '__main__':
    create_final_dataset()