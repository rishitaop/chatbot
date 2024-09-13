#  sidebar-buttons
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data
# df = pd.read_excel('chatbot-1.xlsx')

# # Ensure all required columns are present
# for col in ['Category', 'Subcategory', 'Questions', 'Answers', 'Links']:
#     if col not in df.columns:
#         df[col] = ''

# df['Category'] = df['Category'].fillna('')
# df['Subcategory'] = df['Subcategory'].fillna('')
# df['Questions'] = df['Questions'].fillna('')
# df['Answers'] = df['Answers'].fillna('')
# df['Links'] = df['Links'].fillna('')

# df['Combined'] = (df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# # Create TF-IDF vectorizer and matrix
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match_score = cosine_similarities[best_match_index]
    
#     # Set a threshold to filter out irrelevant matches
#     if best_match_score < 0.3:  # Adjust this threshold as needed
#         return "Couldn't find what you are looking for, try again."
    
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = best_match['Answers']
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# def get_related_links(user_input):
#     related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
#                       (df['Subcategory'].str.contains(user_input, case=False))]
#     if not related_rows.empty:
#         unique_links = related_rows[['Questions', 'Links', 'Answers']].drop_duplicates()
#         if len(unique_links) == 1:
#             row = unique_links.iloc[0]
#             return f"Please use this link to get the answer to your query: [{row['Links']}]({row['Answers']})"
#         else:
#             links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in unique_links.iterrows()])
#             return f"This is what I found. Were you looking for any of these?\n{links}"
#     else:
#         return None

# # Initialize session state for messages
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

# # Title
# st.title("Daas Chatbot 💬")

# # Custom CSS for sidebar
# st.markdown("""
#     <style>
#     [data-testid="stSidebar"] {
#         background-color: #4F2170;  /* Dark Purple */
#     }
#     [data-testid="stSidebar"] img {
#         margin-top: -30px;  /* Move logo slightly up */
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Update sidebar logo
# logo_path = "logo1.png"  # Change logo to logo1
# st.sidebar.image(logo_path, width=280)

# # Sidebar buttons
# categories = df['Category'].unique().tolist()
# num_cols = 2
# cols = st.sidebar.columns(num_cols)

# for idx, category in enumerate(categories):
#     col = cols[idx % num_cols]
#     if col.button(category):
#         related_questions = df[df['Category'] == category][['Questions', 'Links', 'Answers']].drop_duplicates()
#         questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in related_questions.iterrows()])
#         st.session_state.messages.append({"role": "assistant", "content": f"Category: {category}\n\n{questions_and_links}"})

# # Display chat messages
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])

# # Handle user input
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
    
#     related_links = get_related_links(prompt)
#     if related_links:
#         answer = related_links
#     else:
#         answer = get_best_match(prompt)
    
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").markdown(answer)




# # sidebar using radio buttons
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# df = pd.read_excel('chatbot-1.xlsx')

# for col in ['Category', 'Subcategory', 'Questions', 'Answers', 'Links']:
#     if col not in df.columns:
#         df[col] = ''

# df['Category'] = df['Category'].fillna('')
# df['Subcategory'] = df['Subcategory'].fillna('')
# df['Questions'] = df['Questions'].fillna('')
# df['Answers'] = df['Answers'].fillna('')
# df['Links'] = df['Links'].fillna('')

# df['Combined'] = (df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = best_match['Answers']
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# def get_related_links(user_input):
#     related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
#                       (df['Subcategory'].str.contains(user_input, case=False))]
#     if not related_rows.empty:
#         unique_links = related_rows[['Questions', 'Links', 'Answers']].drop_duplicates()
#         if len(unique_links) == 1:
#             row = unique_links.iloc[0]
#             return f"Please use this link to get the answer to your query: [{row['Links']}]({row['Answers']})"
#         else:
#             links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in unique_links.iterrows()])
#             return f"This is what I found. Were you looking for any of these?\n{links}"
#     else:
#         return None

# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
# if 'last_category' not in st.session_state:
#     st.session_state['last_category'] = None

# st.title("Daas Chatbot 💬")
# logo_path = "logo.png"  
# st.sidebar.image(logo_path, width=280)

# categories = df['Category'].unique().tolist()
# categories.insert(0, "Select a Category")

# selected_category = st.sidebar.radio("", options=categories)

# if selected_category and selected_category != "Select a Category" and selected_category != st.session_state['last_category']:
#     st.session_state['last_category'] = selected_category
#     related_questions = df[df['Category'] == selected_category][['Questions', 'Links', 'Answers']].drop_duplicates()
#     questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in related_questions.iterrows()])
#     st.session_state.messages.append({"role": "assistant", "content": f"Category: {selected_category}\n\n{questions_and_links}"})

# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])

# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
    
#     related_links = get_related_links(prompt)
#     if related_links:
#         answer = related_links
#     else:
#         answer = get_best_match(prompt)
    
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").markdown(answer)




# no result found..?
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# df = pd.read_excel('chatbot-1.xlsx')

# for col in ['Category', 'Subcategory', 'Questions', 'Answers', 'Links']:
#     if col not in df.columns:
#         df[col] = ''

# df['Category'] = df['Category'].fillna('')
# df['Subcategory'] = df['Subcategory'].fillna('')
# df['Questions'] = df['Questions'].fillna('')
# df['Answers'] = df['Answers'].fillna('')
# df['Links'] = df['Links'].fillna('')

# df['Combined'] = (df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])


# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match_score = cosine_similarities[best_match_index]
    
  
#     if best_match_score < 0.3: 
#         return "Couldn't find what you are looking for. Please try again!"
    
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = best_match['Answers']
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"


# def get_related_links(user_input):
#     related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
#                       (df['Subcategory'].str.contains(user_input, case=False))]
#     if not related_rows.empty:
#         unique_links = related_rows[['Questions', 'Links', 'Answers']].drop_duplicates()
#         if len(unique_links) == 1:
#             row = unique_links.iloc[0]
#             return f"Please use this link to get the answer to your query: [{row['Links']}]({row['Answers']})"
#         else:
#             links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in unique_links.iterrows()])
#             return f"This is what I found. Were you looking for any of these?\n{links}"
#     else:
#         return None


# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
# if 'last_category' not in st.session_state:
#     st.session_state['last_category'] = None


# st.title("Daas Chatbot 💬")
# logo_path = "logo.png"  
# st.sidebar.image(logo_path, width=280)

# categories = df['Category'].unique().tolist()
# categories.insert(0, "Select a Category")

# selected_category = st.sidebar.radio("", options=categories)

# # Display related questions and links if a category is selected
# if selected_category and selected_category != "Select a Category" and selected_category != st.session_state['last_category']:
#     st.session_state['last_category'] = selected_category
#     related_questions = df[df['Category'] == selected_category][['Questions', 'Links', 'Answers']].drop_duplicates()
#     questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in related_questions.iterrows()])
#     st.session_state.messages.append({"role": "assistant", "content": f"Category: {selected_category}\n\n{questions_and_links}"})

# # Display chat history
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])

# # Handle user input and chatbot response
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
    
#     related_links = get_related_links(prompt)
#     if related_links:
#         answer = related_links
#     else:
#         answer = get_best_match(prompt)
    
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").markdown(answer)


# #UI changes try
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load data from Excel
# df = pd.read_excel('chatbot-1.xlsx')

# # Ensure columns exist and handle missing data
# for col in ['Category', 'Subcategory', 'Questions', 'Answers', 'Links']:
#     if col not in df.columns:
#         df[col] = ''

# df['Category'] = df['Category'].fillna('')
# df['Subcategory'] = df['Subcategory'].fillna('')
# df['Questions'] = df['Questions'].fillna('')
# df['Answers'] = df['Answers'].fillna('')
# df['Links'] = df['Links'].fillna('')

# # Combine relevant columns for text matching
# df['Combined'] = (df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# # Function to get the best match based on user input
# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match_score = cosine_similarities[best_match_index]
    
#     # Set a higher threshold to filter out irrelevant matches
#     if best_match_score < 0.3:  # Increased threshold for better relevance
#         return "Couldn't find what you are looking for, try again."
    
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = best_match['Answers']
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# # Function to get related links based on user input
# def get_related_links(user_input):
#     related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
#                       (df['Subcategory'].str.contains(user_input, case=False))]
#     if not related_rows.empty:
#         unique_links = related_rows[['Questions', 'Links', 'Answers']].drop_duplicates()
#         if len(unique_links) == 1:
#             row = unique_links.iloc[0]
#             return f"Please use this link to get the answer to your query: [{row['Links']}]({row['Answers']})"
#         else:
#             links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in unique_links.iterrows()])
#             return f"This is what I found. Were you looking for any of these?\n{links}"
#     else:
#         return None

# # Inject custom CSS for customizations
# st.markdown(
#     """
#     <style>
#     /* Sidebar background color */
#     [data-testid="stSidebar"] {
#         background-color: #4F2170;  /* Dark Purple */
#     }
#     /* Sidebar text color */
#     [data-testid="stSidebar"] * {
#         color: white;  /* Orange color */
#         font-size: 18px;
#     }
#     /* Logo positioning */
#     [data-testid="stSidebar"] img {
#         margin-top: -30px;  /* Move logo slightly up */
#     }
#     /* Centering and styling the title */
#     .title-container {
#         text-align: center;
#         margin-top: -50px;
#     }
#     .title-container h1 {
#         color: #4F2170;
#         font-size: 3em;
#     }

    
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Streamlit session state initialization
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
# if 'last_category' not in st.session_state:
#     st.session_state['last_category'] = None

# # UI Elements
# st.markdown(
#     """
#     <div class="title-container">
#         <h1>Daas Chatbot 💬</h1>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

# logo_path = "logo1.png"  
# st.sidebar.image(logo_path, width=280)

# categories = df['Category'].unique().tolist()
# categories.insert(0, "Select a Category")

# selected_category = st.sidebar.radio("Select a Category",options=categories,label_visibility="collapsed" )

# # Display related questions and links if a category is selected
# if selected_category and selected_category != "Select a Category" and selected_category != st.session_state['last_category']:
#     st.session_state['last_category'] = selected_category
#     related_questions = df[df['Category'] == selected_category][['Questions', 'Links', 'Answers']].drop_duplicates()
#     questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in related_questions.iterrows()])
#     st.session_state.messages.append({"role": "assistant", "content": f"Category: {selected_category}\n\n{questions_and_links}"})

# # Display chat history
# for msg in st.session_state.messages:
#     if msg["role"] == "user":
#         st.chat_message("user").write(msg["content"])
#     else:
#         st.chat_message("assistant").write(msg["content"])

# # Handle user input and chatbot response
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)
    
#     related_links = get_related_links(prompt)
#     if related_links:
#         answer = related_links
#     else:
#         answer = get_best_match(prompt)
    
#     st.session_state.messages.append({"role": "assistant", "content": answer})
#     st.chat_message("assistant").markdown(answer)


#more coe
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data from Excel
df = pd.read_excel('chatbot-1.xlsx')

# Ensure columns exist and handle missing data
for col in ['Main', 'Category', 'Subcategory', 'Questions', 'Answers', 'Links']:
    if col not in df.columns:
        df[col] = ''

df['Main'] = df['Main'].fillna('')
df['Category'] = df['Category'].fillna('')
df['Subcategory'] = df['Subcategory'].fillna('')
df['Questions'] = df['Questions'].fillna('')
df['Answers'] = df['Answers'].fillna('')
df['Links'] = df['Links'].fillna('')

# Combine relevant columns for text matching
df['Combined'] = (df['Main'] + ' ' + df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# Function to get the best match based on user input
def get_best_match(user_input):
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    best_match_score = cosine_similarities[best_match_index]
    
    # Set a higher threshold to filter out irrelevant matches
    if best_match_score < 0.3:
        return "Couldn't find what you are looking for, try again."
    
    best_match = df.iloc[best_match_index]
    link_name = best_match['Links']
    link_url = best_match['Answers']
    return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# Function to get related links based on user input
def get_related_links(user_input):
    related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
                      (df['Subcategory'].str.contains(user_input, case=False))]
    if not related_rows.empty:
        unique_links = related_rows[['Questions', 'Links', 'Answers']].drop_duplicates()
        if len(unique_links) == 1:
            row = unique_links.iloc[0]
            return f"Please use this link to get the answer to your query: [{row['Links']}]({row['Answers']})"
        else:
            links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in unique_links.iterrows()])
            return f"This is what I found. Were you looking for any of these?\n{links}"
    else:
        return None

# Inject custom CSS for customizations
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #4F2170;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    /* Increase font size for main COE radio buttons */
    .main-category label {
        font-size: 22px;
        font-weight: bold;
    }
    /* Reduce font size for subcategories */
    .subcategory label {
        font-size: 20px;
    }
    /* Add white line between main categories and subcategories */
    .divider-line {
        border-top: 2px solid white;
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit session state initialization
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
if 'last_category' not in st.session_state:
    st.session_state['last_category'] = None
if 'last_main' not in st.session_state:
    st.session_state['last_main'] = None

# UI Elements
st.markdown(
    """
    <div class="title-container">
        <h1>Daas Chatbot 💬</h1>
    </div>
    """,
    unsafe_allow_html=True
)

logo_path = "logo1.png"  
st.sidebar.image(logo_path, width=280)
st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# Get unique main categories and add the default option
main_categories = ['Select a COE'] + df['Main'].unique().tolist()

# Radio buttons for main categories with a default "Select a COE" option
st.sidebar.markdown('<div class="main-category">', unsafe_allow_html=True)
selected_main = st.sidebar.radio("", options=main_categories)
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Add a divider line
st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# If a valid main category is selected (not "Select a COE"), show subcategory radio buttons
if selected_main and selected_main != "Select a COE":
    subcategories = ['Common Categories'] + df[df['Main'] == selected_main]['Category'].unique().tolist()
    
    # Display subcategories as radio buttons
    st.sidebar.markdown('<div class="subcategory">', unsafe_allow_html=True)
    selected_category = st.sidebar.radio("", options=subcategories)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Display related questions and links if a valid subcategory is selected
    if selected_category and selected_category != "Common Categories" and selected_category != st.session_state['last_category']:
        st.session_state['last_category'] = selected_category
        related_questions = df[(df['Main'] == selected_main) & (df['Category'] == selected_category)][['Questions', 'Links', 'Answers']].drop_duplicates()
        questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({row['Answers']})" for _, row in related_questions.iterrows()])
        st.session_state.messages.append({"role": "assistant", "content": f"Category: {selected_category}\n\n{questions_and_links}"})

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# Handle user input and chatbot response
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    related_links = get_related_links(prompt)
    if related_links:
        answer = related_links
    else:
        answer = get_best_match(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
