import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure Charts directory exists
if not os.path.exists('Charts'):
    os.makedirs('Charts')

# Read the CSV file
df = pd.read_csv('Data/BBC_train_full_tokens.csv', names=['category', 'text'])

def create_visualization_collage():
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('BBC News Data Analysis', fontsize=16, y=0.95)

    # 1. Category Distribution (top left)
    category_counts = df['category'].value_counts()
    sns.barplot(x=category_counts.index, y=category_counts.values, ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of News Categories')
    axes[0, 0].set_xlabel('Category')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Text Length Distribution (top right)
    df['text_length'] = df['text'].str.len()
    sns.boxplot(x='category', y='text_length', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Text Length Distribution by Category')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Text Length')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Dot Plot of Text Length by Category (bottom left)
    sns.stripplot(x='category', y='text_length', data=df, ax=axes[1, 0], jitter=True, alpha=0.5)
    axes[1, 0].set_title('Dot Plot of Text Length by Category')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Text Length')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # 4. Average Text Length by Category (bottom right)
    avg_length = df.groupby('category')['text_length'].mean().sort_values(ascending=False)
    sns.barplot(x=avg_length.index, y=avg_length.values, ax=axes[1, 1])
    axes[1, 1].set_title('Average Text Length by Category')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Average Length')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('Charts/bbc_news_analysis_collage.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_all_charts():
    print("Generating visualization collage...")
    create_visualization_collage()
    print("Collage has been saved as 'bbc_news_analysis_collage.png' in the Charts folder.")

if __name__ == "__main__":
    generate_all_charts()
