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
            background-color: #4F2170; 
            color: white; 
            border: 2px solid white; 
            border-radius: 18px;
            padding: 10px 20px; 
            font-size: 16px;
            
        }
        button[data-testid="baseButton-secondary"]:hover {
            background-color: #3b1a56; 
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
    st.experimental_rerun()  # Reload the page
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
            st.experimental_rerun()  # Rerun to show processes for the selected phase
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
            st.experimental_rerun()  # Rerun to show details for the selected process
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
