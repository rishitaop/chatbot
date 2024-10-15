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
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# df = pd.read_excel('chatbot-1.xlsx')

# for col in ['Main', 'Category', 'Subcategory', 'Questions', 'Answers', 'Links', 'Page']:
#     if col not in df.columns:
#         df[col] = ''

# df.fillna('', inplace=True)  

# df['Combined'] = (df['Main'] + ' ' + df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# def construct_link(base_link, page):
#     if page:
#         return f"{base_link}#page={page}"
#     return base_link

# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match_score = cosine_similarities[best_match_index]
    
#     if best_match_score < 0.3:
#         return "Couldn't find what you are looking for, try again."
    
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = construct_link(best_match['Answers'], best_match['Page'])
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# # Function to get related links based on user input
# def get_related_links(user_input):
#     related_rows = df[(df['Category'].str.contains(user_input, case=False)) | 
#                       (df['Subcategory'].str.contains(user_input, case=False))]
#     if not related_rows.empty:
#         unique_links = related_rows[['Questions', 'Links', 'Answers', 'Page']].drop_duplicates()
#         if len(unique_links) == 1:
#             row = unique_links.iloc[0]
#             return f"Please use this link to get the answer to your query: [{row['Links']}]({construct_link(row['Answers'], row['Page'])})"
#         else:
#             links = "\n".join([
#                 f"* {row['Questions']}: [{row['Links']}]({construct_link(row['Answers'], row['Page'])})" 
#                 for _, row in unique_links.iterrows()
#             ])
#             return f"This is what I found. Were you looking for any of these?\n{links}"
#     else:
#         return None


# st.markdown(
#     """
#     <style>
#     [data-testid="stSidebar"] {
#         background-color: #4F2170;
#     }
#     [data-testid="stSidebar"] * {
#         color: white;
#     }
#     .main-category label {
#         font-size: 22px;
#         font-weight: bold;
#     }
#     .subcategory label {
#         font-size: 20px;
#     }
#     .divider-line {
#         border-top: 2px solid white;
#         margin: 5px 0;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
# if 'last_category' not in st.session_state:
#     st.session_state['last_category'] = None
# if 'last_main' not in st.session_state:
#     st.session_state['last_main'] = None

# # UI 
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
# st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# main_categories = ['Select a COE'] + df['Main'].unique().tolist()

# st.sidebar.markdown('<div class="main-category">', unsafe_allow_html=True)
# selected_main = st.sidebar.radio("", options=main_categories)
# st.sidebar.markdown('</div>', unsafe_allow_html=True)

# st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# if selected_main and selected_main != "Select a COE":
#     subcategories = ['Common Categories'] + df[df['Main'] == selected_main]['Category'].unique().tolist()
#     st.sidebar.markdown('<div class="subcategory">', unsafe_allow_html=True)
#     selected_category = st.sidebar.radio("", options=subcategories)
#     st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
#     if selected_category and selected_category != "Common Categories" and selected_category != st.session_state['last_category']:
#         st.session_state['last_category'] = selected_category
#         related_questions = df[(df['Main'] == selected_main) & (df['Category'] == selected_category)][['Questions', 'Links', 'Answers', 'Page']].drop_duplicates()
#         questions_and_links = "\n".join([f"* {row['Questions']}: [{row['Links']}]({construct_link(row['Answers'], row['Page'])})" for _, row in related_questions.iterrows()])
#         st.session_state.messages.append({"role": "assistant", "content": f"Category: {selected_category}\n\n{questions_and_links}"})

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
#https://teams.mdlz.com/:b:/r/sites/dadocumentrepository/Shared%20Documents/Data%20Domain%20Experts/DaaS%20PPTs/May%202024%20Visit%20Final%20versions/DaaS%20Day%20in%20life%20-%20Functional%20Towers.pdf?action=embedview#page=3


#guided tour
# import streamlit as st
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load Excel files
# df = pd.read_excel('chatbot-1.xlsx')
# guided_df = pd.read_excel('Guided_Tour.xlsx')

# # Ensure columns exist in the dataframe
# for col in ['Main', 'Category', 'Subcategory', 'Questions', 'Answers', 'Links', 'Page']:
#     if col not in df.columns:
#         df[col] = ''
# df.fillna('', inplace=True)
# df['Combined'] = (df['Main'] + ' ' + df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)

# # TF-IDF Vectorization
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# # Helper function to construct links
# def construct_link(base_link, page):
#     return f"{base_link}#page={page}" if page else base_link

# # Get the best match for user input
# def get_best_match(user_input):
#     user_tfidf = vectorizer.transform([user_input])
#     cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
#     best_match_index = cosine_similarities.argmax()
#     best_match_score = cosine_similarities[best_match_index]
#     if best_match_score < 0.3:
#         return "Couldn't find what you are looking for, try again."
#     best_match = df.iloc[best_match_index]
#     link_name = best_match['Links']
#     link_url = construct_link(best_match['Answers'], best_match['Page'])
#     return f"Please use this link to get the answer to your query: [{link_name}]({link_url})"

# # Sidebar options
# st.sidebar.header("Select Mode")
# main_options = ["Original Chatbot", "Guided Tour"]
# selected_mode = st.sidebar.radio("", main_options)

# # CSS Styling for buttons and icons
# st.markdown("""
#     <style>
#     [data-testid="stSidebar"] {
#         background-color: #4F2170;
#     }
#     .sidebar-button {
#         background-color: white; /* Button color */
#         color: #E6AF23; /* Text color on buttons */
#         border-radius: 8px; /* Rounded corners */
#         padding: 10px; /* Padding for buttons */
#         font-weight: bold; /* Bold text */
#         text-align: left; /* Align text to the left */
#         display: block; /* Block display for stacking */
#         margin: 5px 0; /* Spacing between buttons */
#     }
#     .sidebar-button:hover {
#         background-color: #FFB300; /* Hover color for buttons */
#     }
#     [data-testid="stSidebar"] * {
#         color: white; /* Keeping other sidebar text in white */
#     }
#     .title-container {
#         text-align: center;
#         color: #4F2170;
#         font-weight: bold;
#     }
#     .horizontal-container {
#         display: flex;
#         justify-content: center;
#         align-items: center;
#         gap: 20px;
#         flex-wrap: nowrap;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Main Title
# st.markdown("<div class='title-container'><h1>Daas Chatbot 💬</h1></div>", unsafe_allow_html=True)

# # Initialize session state variables
# if 'messages' not in st.session_state:
#     st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
# if 'selected_phase' not in st.session_state:
#     st.session_state['selected_phase'] = None

# # Display conversation history for the original chatbot mode
# if selected_mode == "Original Chatbot":
#     for msg in st.session_state.messages:
#         if msg["role"] == "user":
#             st.chat_message("user").write(msg["content"])
#         else:
#             st.chat_message("assistant").write(msg["content"])

#     # Accept user input
#     user_input = st.chat_input("Your message here...")
#     if user_input:
#         st.session_state.messages.append({"role": "user", "content": user_input})

#         # Fetch and display the chatbot response
#         response = get_best_match(user_input)
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         st.chat_message("assistant").write(response)

# # Guided Tour Section
# elif selected_mode == "Guided Tour":
#     st.write("What Phase are you currently in?")

#     # Horizontal button layout for Guided Tour steps
#     steps = {
#         'Demand Intake Process': ('🔍', '#9400D3'),  # Violet
#         'Estimation/Analysis': ('📊', '#4B0082'),  # Indigo
#         'Design': ('✏️', '#0000FF'),  # Blue
#         'Development': ('💻', '#008000'),  # Green
#         'Testing': ('🧪', '#FFFF00'),  # Yellow
#         'Go live': ('🚀', '#FFA500'),  # Orange
#         'HyperCare': ('⚙️', '#FF0000')  # Red
#     }

#     # Render the horizontal buttons only if no phase is selected
#     if not st.session_state['selected_phase']:
#         st.markdown('<div class="horizontal-container">', unsafe_allow_html=True)
#         for step, (icon, color) in steps.items():
#             if st.button(f"{icon} {step}", key=step, use_container_width=True):
#                 st.session_state['selected_phase'] = step  # Set selected phase
#                 # Immediately remove buttons after selecting a phase
#                 st.experimental_rerun()  # Refresh the app to remove the main buttons
#         st.markdown('</div>', unsafe_allow_html=True)

#     # Display sidebar only if a phase is selected
#     if st.session_state['selected_phase']:
#         # Add matching buttons in the sidebar
#         st.sidebar.subheader("Quick Navigation")
#         for step, (icon, color) in steps.items():
#             if st.sidebar.button(f"{icon} {step}", key=f"sidebar_{step}", help=f"Navigate to {step}"):
#                 st.session_state['selected_phase'] = step  # Set the selected phase on sidebar button click

#         # Remove main buttons after phase selection
#         st.markdown("<style> .horizontal-container { display: none; } </style>", unsafe_allow_html=True)

#         # Display detailed steps based on phase selection
#         selected_phase = st.session_state['selected_phase']
#         st.write(f"### {selected_phase} Phase Details")

#         # Process selection for current step
#         phase_steps = guided_df[guided_df['Phases'] == selected_phase]['Process'].unique().tolist()
#         current_step = st.radio("Which step are you currently on?", options=["Choose Step"] + phase_steps, key=f"current_step_{selected_phase}")

#         # Display current step details only if a valid step is chosen
#         if current_step and current_step != "Choose Step":
#             current_step_data = guided_df[(guided_df['Phases'] == selected_phase) & (guided_df['Process'] == current_step)]
#             for _, row in current_step_data.iterrows():
#                 st.write(f"**Process**: {row['Process']}")
#                 st.write(f"**Document**: [Link]({row['Document']})")
#                 st.write(f"**SPOC**: {row['SPOC']}")
#                 st.write("---")

#             # Suggested Next Steps
#             next_steps = phase_steps[phase_steps.index(current_step) + 1:]
#             if next_steps:
#                 st.write(f"**What do you need to do next?**")
#                 for step in next_steps:
#                     step_data = guided_df[(guided_df['Phases'] == selected_phase) & (guided_df['Process'] == step)]
#                     st.write(f"**Next Step**: {step}")
#                     for _, row in step_data.iterrows():
#                         st.write(f"**Document**: [Link]({row['Document']})")
#                         st.write(f"**SPOC**: {row['SPOC']}")
#                         st.write("---")

# new version chatbot
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Excel files for the main chatbot and guided tour data
chatbot_file = 'chatbot-1.xlsx'
guided_tour_file = 'Guided_Tour.xlsx'
df = pd.read_excel(chatbot_file)
guided_df = pd.read_excel(guided_tour_file, sheet_name='Main')
faq_df = pd.read_excel(chatbot_file, sheet_name='FAQ')  # Load the FAQ data

# Ensure required columns exist
for col in ['Main', 'Category', 'Subcategory', 'Questions', 'Answers', 'Links', 'Page']:
    if col not in df.columns:
        df[col] = ''
df.fillna('', inplace=True)

# Initialize TF-IDF Vectorizer for similarity matching
df['Combined'] = (df['Main'] + ' ' + df['Category'] + ' ' + df['Subcategory'] + ' ' + df['Questions']).astype(str)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Combined'])

# Helper function to construct links
def construct_link(base_link, page):
    if page:
        return f"{base_link}#page={page}"
    return base_link


def get_links_by_category(selected_category):
    """Retrieve unique links from the Excel file based on exact category matching."""
    
    # Filter rows matching the selected category
    matching_rows = df[df['Category'].str.strip().str.lower() == selected_category.strip().lower()]

    if matching_rows.empty:
        return "No links found for the selected category."

    # Collect and format unique links
    results = []
    seen_links = set()  # To track unique URLs

    for _, row in matching_rows.iterrows():
        link_name = row['Links'].strip()
        link_url = construct_link(row['Answers'].strip(), row['Page'])

        # Ensure link name and URL are valid and unique
        if link_name and link_url and link_url not in seen_links:
            seen_links.add(link_url)
            result_entry = f"- [{link_name}]({link_url})"
            results.append(result_entry)

    # Join results into a single string with bullet points
    if results:
        return "\n\n".join(results)
    else:
        return "No unique links available for the selected category."


# Function to get all relevant matches for user input
def get_all_matches(user_input, category_filter=None):
    """Find matching links based on user input and optional category filter."""
    user_tfidf = vectorizer.transform([user_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Filter matches with similarity above a dynamic threshold (0.25).
    matched_indices = [i for i, score in enumerate(cosine_similarities) if score >= 0.25]

    if not matched_indices:
        return "Couldn't find what you are looking for, try again."

    # Collect and format all matched results with unique links
    results = []
    seen_links = set()  # To track unique links

    for idx in matched_indices:
        match = df.iloc[idx]

        # Normalize category names to avoid case or whitespace mismatch
        if category_filter and match['Category'].strip().lower() != category_filter.strip().lower():
            continue  # Skip if the category doesn't match

        link_name = match['Links'].strip()  # Clean link name
        link_url = construct_link(match['Answers'].strip(), match['Page'])  # Ensure proper URL formatting

        # Ensure the link name and URL are valid
        if link_name and link_url:
            cleaned_link_name = link_name.split('(')[0].strip()  # Remove notes inside parentheses
            cleaned_link_url = link_url.split(' ')[0]  # Take only the first part of the URL

            # Add only unique links to the results
            if cleaned_link_url not in seen_links:
                seen_links.add(cleaned_link_url)
                result_entry = f"- [{cleaned_link_name}]({cleaned_link_url})"
                results.append(result_entry)

    # Join all results into a single string with bullet points
    if results:
        return "This is what I found, were you looking for any of these?:\n\n" + "\n\n".join(results)
    else:
        return "Couldn't find any unique links for your query."


# CSS styling for the sidebar and buttons
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #4F2170;
    }
    [data-testid="stSidebar"] * {
        color: white;
    }
    .title-container {
        text-align: center;
        color: #4F2170;
        font-weight: bold;
     }
    button[data-testid="baseButton-secondary"] {
        background-color: #4F2170; !important
        color: white; !important
        border: 2px solid white; !important
        border-radius: 18px; !important
        padding: 10px 20px; !important
        font-size: 16px; !important
            
        }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #3b1a56; !important
        }
     
    .horizontal-container {
        display: flex;
        # justify-content: center;
        # align-items: center;
        # gap: 20px;
        # flex-wrap: nowrap;
        background-color: #4F2170;
    }
    
    .phase-button {
        background-color: #FFB300;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        margin: 5px;
        width: 200px;
        text-align: center;
        cursor: pointer;
        display: inline-block;
    }
    .phase-button:hover {
        background-color: #FF4500;
    }
    .divider-line {
        border-top: 2px solid white;
        margin: 2px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title-container'><h1>Daas Chatbot 💬</h1></div>", unsafe_allow_html=True)

# User input for chatbot
user_input = st.chat_input("Your Question")
if user_input:
    results = get_all_matches(user_input)  # Get results based on user input
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": results})


# Sidebar with logo and COE selection
logo_path = "logo1.png"  # Update the logo path if necessary
st.sidebar.image(logo_path, width=280)
st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

if st.sidebar.button("🏠 Home", key="home_button"):
    st.session_state.clear()  # Clear all session state
    # st.experimental_rerun()  # Reload the page
st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# COE selection
main_categories = df['Main'].unique().tolist()

# Loop through each COE to create an expander for categories
for main_category in main_categories:
    with st.sidebar.expander(main_category, expanded=False):  # Create an expander for each COE
        categories = df[df['Main'] == main_category]['Category'].unique().tolist()
        selected_category = st.radio(f"Select a Category for {main_category}", options=["Select a Category"] + list(categories), 
                                        key=main_category)
        
        # Only proceed if no user input is provided and a category is selected
        if not user_input and selected_category != "Select a Category":
            links_results = get_links_by_category(selected_category)  # Use the category as input to get links
            st.session_state.messages.append({"role": "user", "content": selected_category})
            st.session_state.messages.append({"role": "assistant", "content": links_results})

# Sidebar buttons for Home and FAQs
st.sidebar.markdown('<div class="horizontal-container">', unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'selected_phase' not in st.session_state:
    st.session_state.selected_phase = None
if 'selected_process' not in st.session_state:
    st.session_state.selected_process = None
if 'project_day_clicked' not in st.session_state:
    st.session_state.project_day_clicked = False

st.sidebar.markdown('<div class="divider-line"></div>', unsafe_allow_html=True)

# Button for Project Day In Life
if st.sidebar.button("📋 Project Day In Life", key="project_day_button"):
    st.session_state.clear()  # Clear all session state
    # st.experimental_rerun() 
    st.session_state.project_day_clicked = True  # Set flag indicating the button was clicked
    st.session_state.selected_phase = None  # Reset phase selection
    st.session_state.selected_process = None  # Reset process selection
    st.session_state.messages = []  # Clear chat history

# Show phases only if the button has been clicked
if st.session_state.project_day_clicked:
    st.write("### What Phase are you currently in?")
    phases = guided_df['Phases'].unique().tolist()
    phase_emojis = {
        'Pre Discovery': '🔍',
        'Discovery & Analysis': '📊',
        'Design': '✏️',
        'Build': '💻',
        'Testing': '🧪',
        'Hyper Care': '🚀',
        'Support': '⚙️'
    }
    st.markdown('<div class="horizontal-container">', unsafe_allow_html=True)
    for phase in phases:
        emoji = phase_emojis.get(phase, '')
        if st.button(f"{emoji} {phase}", key=phase, use_container_width=True):
            st.session_state.selected_phase = phase
            # st.experimental_rerun()  # Rerun to show processes for the selected phase
    st.markdown('</div>', unsafe_allow_html=True)

# Process selection based on phase
if st.session_state.selected_phase:
    selected_phase = st.session_state.selected_phase
    st.session_state.messages = []  # Clear chat history
    st.write(f"### {selected_phase} phase: What process do you need help with?")
    processes = guided_df[guided_df['Phases'] == selected_phase]['Process'].unique().tolist()
    st.markdown('<div class="horizontal-container">', unsafe_allow_html=True)
    for process in processes:
        if st.button(f"{process}", key=f"process_{process}"):
            st.session_state.selected_process = process
                    # st.experimental_rerun()  # Rerun to show details for the selected process
    st.markdown('</div>', unsafe_allow_html=True)

# Show details based on the selected process
if st.session_state.selected_process:
    selected_process = st.session_state.selected_process
    st.write(f"### Details for **{selected_process}** process:")

    # Filter the DataFrame for the selected phase and process
    process_data = guided_df[
        (guided_df['Phases'] == selected_phase) & 
        (guided_df['Process'] == selected_process)
    ]

    # Iterate over the matching rows and display relevant information
    for _, row in process_data.iterrows():
        # Display the Document link with its label
        document_name = row['Document']
        document_url = row['Document_link']
        st.write(f"**Document**: [{document_name}]({document_url})" if document_url else f"**Document**: {document_name}")

        # Display the SPOC link with its label
        spoc_name = row['SPOC']
        spoc_url = row['SPOC_link']
        st.write(f"**SPOC**: [{spoc_name}]({spoc_url})" if spoc_url else f"**SPOC**: {spoc_name}")

        # Display the SPOC link with its label
        raci_name = row['RACI']
        raci_url = row['RACI_link']
        st.write(f"**RACI**: [{raci_name}]({raci_url})" if raci_url else f"**RACI**: {raci_name}")
        st.write("---")  # Separator for multiple entries
        st.session_state.clear()  # Clear all session state


    # Clear chat history after displaying the process details
    st.session_state.messages = []


# Button for Home


# Button for FAQs
if st.sidebar.button("📚 FAQs", key="faq_button"):

    st.session_state.show_faqs = True

    # st.session_state.selected_phase = None  # Clear selected phase to reset Project Day In Life
    st.session_state.messages = []  # Clear chat historys

if st.session_state.get("show_faqs"):
    st.session_state.clear()  # Clear all session state


    st.write("### Frequently Asked Questions")
    
    # Display all FAQs
    for _, row in faq_df.iterrows():
        question = row['Common_Question']
        link_name = row['Name']
        link_url = row['Ans_Links']
        st.write(f"**Q:** {question}")
        if link_name and link_url:
            st.write(f"- **Link:** [{link_name}]({link_url})")
        st.write("---")  # Divider for questions
        st.session_state.messages = []  # Clear chat history
        st.session_state.clear()  # Clear all session state

   


# Chat messages state
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]
 
# Display chat messages
if st.session_state.messages:
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message("user").markdown(message['content'])
        else:
            st.chat_message("assistant").markdown(message['content'])
