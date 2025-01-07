# %% [markdown]
# # **Predictive Modeling for Rocket Landing Success**  
# ### *A Machine Learning Approach Using SpaceX Falcon 9 Data* 

# %% [markdown]
# ## **Data Wrangling**
# 
# In this stage, we perform data wrangling to clean, transform, and structure raw data, ensuring it is accurate and consistent for analysis and modeling. This process enables us to uncover patterns and relationships within the data, laying the foundation for building reliable supervised machine learning models.
# 
# The dataset includes various records of booster landing outcomes, categorized as follows:
# 
# - **True Ocean**: indicates a successful landing in the ocean.
# - **False Ocean**: indicates a failed landing in the ocean.
# - **True RTLS**: indicates a successful landing on a ground-based landing pad.
# - **False RTLS**: indicates a failed landing on a ground-based landing pad.
# - **True ASDS**: indicates a successful landing on a drone ship.
# - **False ASDS**: indicates a failed landing on a drone ship.
# 

# %% [markdown]
# **Falcon 9 first stage will land successfully**
# 
# ![landing_1.gif](attachment:landing_1.gif)
# 

# %% [markdown]
# **Several examples of an unsuccessful landing are shown here:**
# 
# ![](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DS0701EN-SkillsNetwork/api/Images/crash.gif)

# %% [markdown]
# ### Import Libraries

# %%
import pandas as pd 
import numpy as np

# %% [markdown]
# ### Load the SpaceX dataset data_falcon9.csv

# %%
# Define the file path
file_path = r'C:\Users\cjchavarria\Desktop\Rocket-Landing-Success-ML\01-data-collection-using-SpaceX-API\data_falcon9.csv'

# %%
try:
    # Load the dataset into a DataFrame with optimized memory usage
    df = pd.read_csv(file_path)
    print("Dataset successfully loaded!")
except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# %%
df.head(5)

# %% [markdown]
# ### Identify numerical and categorical columns

# %%
def classify_columns(df):
    # Separate columns based on their data types
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Create a DataFrame for numerical columns with their data types
    numerical_df = pd.DataFrame({
        'Column Name': numerical_columns,
        'Data Type': [df[col].dtype for col in numerical_columns]
    })

    # Create a DataFrame for categorical columns with their data types
    categorical_df = pd.DataFrame({
        'Column Name': categorical_columns,
        'Data Type': [df[col].dtype for col in categorical_columns]
    })

    return numerical_df, categorical_df

# Call the function and display the results
numerical_df, categorical_df = classify_columns(df)

print("Numerical DataFrame: ")
print(numerical_df)
print("\nCategorical DataFrame: ")
print(categorical_df)


# %% [markdown]
# ### Calculate and display the percentage of missing values per attribute

# %%
def calculate_missing_percentage(df):
    #calculate the percentage of missing values
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    # create a DataFrame for better readability
    missing_summary = missing_percentage.reset_index()
    missing_summary.columns = ['Attribute', 'MissingPercentage']

    # filter attributes with missing values and sort them in descending order
    missing_summary = missing_summary[missing_summary["MissingPercentage"] >0]
    missing_summary = missing_summary.sort_values(by = 'MissingPercentage', ascending = False).reset_index(drop = True)

    return missing_summary

# call the function and display the result
missing_data = calculate_missing_percentage(df)
if not missing_data.empty:
    print(missing_data)
else:
    print("No missing values found in the DataFrame")

# %% [markdown]
# #### Calculating and Replacing Missing Values with the Mean
# 
# We calculate the mean of the `PayloadMass` column and replace the missing (NaN) values in that column with the calculated mean.

# %%
# Calculate the mean of the PayloadMass column
mean_payload_mass = df['PayloadMass'].mean()
print(mean_payload_mass)

# %%
# Replace missing values in PayloadMass with the calculated mean using .loc[]
df.loc[:, 'PayloadMass'] = df['PayloadMass'].fillna(mean_payload_mass)

# %% [markdown]
# ### Calculate the number of launches on each site
# The dataset includes multiple SpaceX launch sites, specifically:
# 
# - **`CCAFS LC-40`: Cape Canaveral Space Launch Complex 40**
# - **`VAFB SLC 4E`: Vandenberg Air Force Base Space Launch Complex 4E**
# - **`KSC LC-39A`: Kennedy Space Center Launch Complex 39A**
# 
# Each launch's site is recorded in the `LaunchSite` column.

# %%
df['LaunchSite'].value_counts().to_frame().reset_index()

# %% [markdown]
# #### Calculate the number and occurence of each orbit
# 
# Each launch aims to an dedicated orbit, and here are some common orbit types:

# %%
df['Orbit'].value_counts().to_frame().reset_index()

# %% [markdown]
# ### Satellite Orbits:
# 
# - **LEO (Low Earth Orbit)**: LEO is an Earth-centered orbit with an altitude of 2,000 km (1,200 mi) or less, approximately one-third of the radius of Earth, or with at least 11.25 periods per day (orbital period of 128 minutes or less) and eccentricity less than 0.25. Most manmade objects in outer space are in LEO. [More info](https://en.wikipedia.org/wiki/Low_Earth_orbit)
# 
# - **VLEO (Very Low Earth Orbit)**: VLEO refers to orbits with a mean altitude below 450 km. Operating in VLEO offers advantages for Earth observation as spacecraft are closer to the observation target. [Research on VLEO](https://www.researchgate.net/publication/271499606_Very_Low_Earth_Orbit_mission_concepts_for_Earth_Observation_Benefits_and_challenges)
# 
# - **GTO (Geosynchronous Transfer Orbit)**: A geosynchronous orbit is a high Earth orbit allowing satellites to match Earth's rotation. Located 35,786 km above the equator, this orbit is ideal for weather monitoring, communications, and surveillance. [NASA's description](https://www.space.com/29222-geosynchronous-orbit.html)
# 
# - **SSO (Sun-Synchronous Orbit)**: Also known as a heliosynchronous orbit, SSO is a nearly polar orbit in which the satellite passes over any given point of the planet's surface at the same local mean solar time. [More info](https://en.wikipedia.org/wiki/Sun-synchronous_orbit)
# 
# - **ES-L1 (Lagrange Point 1)**: Lagrange points are locations where the gravitational forces of two large bodies cancel out. L1 is one such point between the Earth and the Sun. [Learn more](https://en.wikipedia.org/wiki/Lagrange_point#L1_point)
# 
# - **HEO (Highly Elliptical Orbit)**: A highly elliptical orbit is one with high eccentricity, typically referring to Earth orbits. [More details](https://en.wikipedia.org/wiki/Highly_elliptical_orbit)
# 
# - **ISS (International Space Station)**: The ISS is a modular space station in low Earth orbit, a collaborative project between NASA, Roscosmos, JAXA, ESA, and CSA. [Learn about ISS](https://en.wikipedia.org/wiki/International_Space_Station)
# 
# - **MEO (Medium Earth Orbit)**: MEO refers to geocentric orbits ranging from 2,000 km to just below geosynchronous orbit (35,786 km). These orbits are commonly at 20,200 km or 20,650 km, with a 12-hour orbital period. [More information](https://en.wikipedia.org/wiki/List_of_orbits)
# 
# - **HEO (High Earth Orbit)**: Geocentric orbits above the altitude of geosynchronous orbit (35,786 km). [More details](https://en.wikipedia.org/wiki/List_of_orbits)
# 
# - **GEO (Geostationary Orbit)**: A circular geosynchronous orbit at 35,786 km above Earth's equator, following Earth's rotation direction. [More about GEO](https://en.wikipedia.org/wiki/Geostationary_orbit)
# 
# - **PO (Polar Orbit)**: A type of orbit where the satellite passes above or near both poles of the planet being orbited, typically Earth. [Learn more](https://en.wikipedia.org/wiki/Polar_orbit)
# 
# #### Some orbits are illustrated in the following plot:
# 

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# #### Calculate the number and occurrence of mission outcome of the orbits
# 

# %%
df['Outcome'].value_counts().to_frame().reset_index()

# %% [markdown]
# In the dataset, the mission outcomes are categorized as follows:
# 
# - **`True Ocean`**: The mission was successfully landed in a specific region of the ocean.
# - **`False Ocean`**: The mission unsuccessfully attempted to land in a specific region of the ocean.
# - **`True RTLS`**: The mission was successfully landed on a ground pad.
# - **`False RTLS`**: The mission unsuccessfully attempted to land on a ground pad.
# - **`True ASDS`**: The mission was successfully landed on a drone ship.
# - **`False ASDS`**: The mission unsuccessfully attempted to land on a drone ship.
# - **`None ASDS`** and **`None None`**: These values represent failed landing attempts.
# 
# The following code iterates through the `landing_outcomes` dictionary and prints the index and outcome:
# 

# %%
landing_outcomes = df['Outcome'].value_counts()

# %%
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)

# %% [markdown]
# We create a set of outcomes where the second stage of the rocket did not land successfully. To do this, we select specific elements from `landing_outcomes.keys()` that correspond to failed landing outcomes.
# 
# The selected indices represent the cases where the landing was unsuccessful. We convert the view of keys into a list so we can index and select the correct keys.

# %%
bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
bad_outcomes

# %% [markdown]
# ### Create a landing outcome label from `Outcome` column
# 
# we create a new variable called `landing_class` based on the values in the `Outcome` column. We assign a label of `0` if the corresponding value in the `Outcome` column is in the `bad_outcomes` set, and a label of `1` otherwise. This will allow us to use `landing_class` as a binary classification label for training supervised models, where `1` indicates a successful landing and `0` indicates a failed landing.

# %%
landing_class = [0 if value in bad_outcomes else 1 for value in df['Outcome']]

# %% [markdown]
# we assign the `landing_class` list, which we created previously, to a new column in the DataFrame called `Class`. This new column will represent the classification variable indicating the outcome of each launch. If the value is `0`, it means the first stage did not land successfully; if the value is `1`, it indicates the first stage landed successfully.

# %%
df['Class'] = landing_class 
df.head(10)

# %%
df['Class'].value_counts().to_frame().reset_index() 

# %% [markdown]
# ### Save the new dataframe in a CSV file:

# %%
df.to_csv(r'C:\Users\cjchavarria\Desktop\Rocket-Landing-Success-ML\02-data-wrangling\data_falcon9_V2.csv' , index = False)


