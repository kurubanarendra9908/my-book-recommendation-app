# streamlit_app_step1.py
import streamlit as st

# App title
st.title("Recommendation System")
# streamlit_app_step1.py

import streamlit as st  # Streamlit for creating the app
import numpy as np      # Numerical computing
import pandas as pd     # Data manipulation
import seaborn as sns   # Visualization
import matplotlib.pyplot as plt  # Plotting
from sklearn.preprocessing import StandardScaler  # Data normalization
from sklearn.metrics.pairwise import cosine_similarity  # Similarity computation

# App title
st.title("📚 Step 1: Importing Libraries for Book Recommendation Engine")

# Description
st.write("""
This Streamlit app demonstrates the first step of our book recommendation engine project — **importing the necessary Python libraries**.
""")

# Show each library with explanation
st.subheader("🔍 Libraries Imported and Their Purpose:")

libraries = {
    "numpy": "Efficient number and array operations",
    "pandas": "Load, clean, and manipulate tabular data (CSV files)",
    "seaborn": "Stylish data plots and charts",
    "matplotlib.pyplot": "Basic plotting and data visualization",
    "StandardScaler (from sklearn)": "Standardizes data (mean=0, std=1)",
    "cosine_similarity (from sklearn)": "Measures similarity between books based on ratings"
}

for lib, desc in libraries.items():
    st.markdown(f"✅ **{lib}**: {desc}")
# Step 2: Load the Dataset 📂

# 📌 Import required libraries
import streamlit as st
import pandas as pd

# 📁 Load datasets from local CSV files
books = pd.read_csv(r"C:\Users\radhi\OneDrive\Desktop\projec\Books.csv", encoding="latin-1")      # Book metadata
ratings = pd.read_csv(r"C:\Users\radhi\OneDrive\Desktop\projec\Ratings.csv")                      # User ratings
users = pd.read_csv(r"C:\Users\radhi\OneDrive\Desktop\projec\Users.csv")                          # User info

# ✅ Display dataset info using Streamlit
st.subheader("📚 Dataset Overview")

# Show dataframes with expanders to save space
with st.expander("View Books Dataset"):
    st.dataframe(books.head())        # Show first 5 rows of books

with st.expander("View Ratings Dataset"):
    st.dataframe(ratings.head())      # Show first 5 rows of ratings

with st.expander("View Users Dataset"):
    st.dataframe(users.head())        # Show first 5 rows of users

# Optional: Show column names or shapes
st.write("✅ **Books shape:**", books.shape)
st.write("✅ **Ratings shape:**", ratings.shape)
st.write("✅ **Users shape:**", users.shape)


# Step 3: Data Cleaning and Preparation 🧹

# 📌 Subheader in Streamlit UI
st.subheader("🧹 Step 3: Data Cleaning and Preparation")

# 1️⃣ Remove duplicate book titles to avoid repetition in recommendations
new_books = books.drop_duplicates(subset='Book-Title')

# 2️⃣ Merge ratings with book data using the 'ISBN' column
ratings_with_name = ratings.merge(new_books, on='ISBN')

# 3️⃣ Drop irrelevant columns such as images and ISBN (optional cleanup)
columns_to_drop = ['ISBN', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L']
ratings_with_name.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 4️⃣ Merge ratings+book data with user data on 'User-ID'
users_ratings_matrix = ratings_with_name.merge(users, on='User-ID')

# 5️⃣ Drop less useful user columns like Location and Age
users_ratings_matrix.drop(['Location', 'Age'], axis=1, inplace=True, errors='ignore')

# ✅ Show a sample of the cleaned and merged dataset
st.markdown("### ✅ Cleaned Ratings Dataset")
st.dataframe(users_ratings_matrix.head())

# 6️⃣ Remove any null values to keep data consistent
users_ratings_matrix.dropna(inplace=True)
st.success("✅ Null values removed (if any).")

# 7️⃣ Keep only users who have rated more than 100 books
user_rating_counts = users_ratings_matrix.groupby('User-ID').count()['Book-Rating']
active_users = user_rating_counts[user_rating_counts > 100].index
filtered_users_ratings = users_ratings_matrix[users_ratings_matrix['User-ID'].isin(active_users)]

# 8️⃣ Keep only books that have received 50 or more ratings
book_rating_counts = filtered_users_ratings.groupby('Book-Title').count()['Book-Rating']
popular_books = book_rating_counts[book_rating_counts >= 50].index
final_users_ratings = filtered_users_ratings[filtered_users_ratings['Book-Title'].isin(popular_books)]

# 9️⃣ Create a pivot table (Book-Title × User-ID) with ratings as values
pivot_table = final_users_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')

# 🔄 Fill missing values with 0 (assume unrated books = 0)
pivot_table.fillna(0, inplace=True)

# 📊 Show pivot table preview
st.markdown("### 📊 Ratings Pivot Table")
st.dataframe(pivot_table.head())

# 🔟 Standardize the pivot table using StandardScaler
scaler = StandardScaler(with_mean=True, with_std=True)
pivot_table_normalized = scaler.fit_transform(pivot_table)

# ✅ Final success message
st.success("🎉 Data cleaning, filtering, pivoting, and normalization complete!")



# Step 4: Model Building 🧠

st.subheader("🧠 Step 4: Model Building")

# 1️⃣ Calculate similarity matrix using cosine similarity
#    This compares how similar two books are based on ratings from users
similarity_score = cosine_similarity(pivot_table_normalized)

st.success("✅ Cosine similarity matrix calculated!")

# 2️⃣ Define the recommendation function
def recommend(book_name):
    """
    Given a book title, return the top 5 most similar books
    based on the cosine similarity score.
    """
    # Check if book exists in the pivot table index
    if book_name not in pivot_table.index:
        return []

    # Get index of the selected book
    index = np.where(pivot_table.index == book_name)[0][0]

    # Get list of books sorted by similarity score (excluding the book itself)
    similar_books = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]  # Skip the first one (same book)

    recommendations = []

    for i, sim in similar_books:
        # Fetch metadata for each similar book
        temp_df = new_books[new_books['Book-Title'] == pivot_table.index[i]]

        # Make sure it's not empty
        if not temp_df.empty:
            title = temp_df['Book-Title'].values[0]
            author = temp_df['Book-Author'].values[0]
            image = temp_df['Image-URL-M'].values[0]
            recommendations.append((title, author, image))

    return recommendations

st.success("✅ Recommendation function is ready to use.")



# Step 5: Model Validation ✅

st.subheader("✅ Step 5: Model Validation")

# 1️⃣ Create a dropdown for the user to select a book title
book_list = pivot_table.index.tolist()
selected_book = st.selectbox("📖 Select a book to get recommendations:", book_list)

# 2️⃣ When the user clicks the button, show recommendations
if st.button("🔍 Recommend Similar Books"):
    recommendations = recommend(selected_book)

    if recommendations:
        st.markdown("### 🔁 Top 5 Recommended Books:")
        for title, author, image in recommendations:
            # Display image and text side-by-side
            st.image(image, width=120)
            st.markdown(f"**{title}**  \n_By {author}_")
    else:
        st.warning("⚠️ Could not find the book or recommendations.")



# 🎨 Step 6: UI Enhancements and Sidebar

# 🏷️ Main title and subtitle
st.title("📚 Book Recommendation Engine")
st.markdown("Get personalized book recommendations using Collaborative Filtering! 🔍")

# 📋 Sidebar: App info
st.sidebar.title("📌 About This App")
st.sidebar.info(
    """
    This app recommends books based on what similar users liked.
    
    - Built with 💡 Cosine Similarity
    - Data: Books.csv, Ratings.csv, Users.csv
    - Filters: Active users & popular books only
    """
)

# 🎯 Bonus: Custom footer or credits
st.markdown("---")
st.markdown("Made with ❤️ by *kuruba Narendra* | Powered by Streamlit")
