from flask import Flask, render_template, request, jsonify
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Prepare the Data (already provided)

# B2B Sales Numbers Dataset
b2b_sales_data = pd.DataFrame({
    'Seller_Company': ['Tech Innovators Inc', 'Green Energy Corp', 'Tech Innovators Inc', 'FinTech Global', 'Green Energy Corp'],
    'Buyer_Company': ['Retail Solutions LLC', 'HealthWorks Ltd', 'SmartHome Devices Co', 'Retail Solutions LLC', 'AgriTech Partners'],
    'Sales_Amount': [50000, 75000, 120000, 60000, 45000],
    'Transaction_Date': ['2024-01-15', '2024-02-20', '2024-03-05', '2024-04-10', '2024-05-18'],
    'Region': ['North America', 'Europe', 'Asia', 'North America', 'Europe'],
    'Payment_Terms': ['Net 30', 'Net 45', 'Net 30', 'Net 60', 'Net 30'],
    'Product_Category': ['Electronics', 'Renewable Energy', 'IoT Devices', 'Financial Software', 'AgriTech'],
    'Future_Prospects': ['Expansion Potential', 'Strong', 'Growing Market', 'Stable', 'Emerging Opportunities']
})

# Financials of a Company Dataset
financial_data = pd.DataFrame({
    'Year': [2021, 2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023],
    'Company': ['Tech Innovators Inc', 'Tech Innovators Inc', 'Tech Innovators Inc', 'Green Energy Corp', 'Green Energy Corp', 'Green Energy Corp',
                'HealthWorks Ltd', 'HealthWorks Ltd', 'HealthWorks Ltd', 'FinTech Global', 'FinTech Global', 'FinTech Global'],
    'Revenue': [500000, 600000, 650000, 450000, 550000, 600000, 700000, 800000, 850000, 600000, 700000, 750000],
    'Net_Income': [50000, 70000, 80000, 40000, 60000, 70000, 80000, 90000, 100000, 50000, 60000, 70000],
    'Total_Assets': [1500000, 1600000, 1700000, 1400000, 1500000, 1600000, 1800000, 1900000, 2000000, 1600000, 1700000, 1800000],
    'Total_Liabilities': [700000, 750000, 800000, 600000, 650000, 700000, 900000, 950000, 1000000, 800000, 850000, 900000],
    'Equity': [800000, 850000, 900000, 800000, 850000, 900000, 900000, 950000, 1000000, 800000, 850000, 900000],
    'Shares_Outstanding': [1000000, 1050000, 1100000, 950000, 1000000, 1050000, 1500000, 1550000, 1600000, 1200000, 1250000, 1300000],
    'Stock_Price': [50, 55, 60, 40, 45, 50, 60, 65, 70, 55, 60, 65],
    'R&D_Expenditure': [100000, 120000, 130000, 80000, 90000, 100000, 120000, 130000, 140000, 90000, 100000, 110000],
    'Future_Growth_Prospects': ['High', 'High', 'Very High', 'Medium', 'High', 'Very High', 'High', 'Very High', 'Very High', 'Medium', 'High', 'High']
})

# ESG Values of a Company Dataset
esg_data = pd.DataFrame({
    'Year': [2021, 2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023, 2021, 2022, 2023],
    'Company': ['Tech Innovators Inc', 'Tech Innovators Inc', 'Tech Innovators Inc', 'Green Energy Corp', 'Green Energy Corp', 'Green Energy Corp',
                'HealthWorks Ltd', 'HealthWorks Ltd', 'HealthWorks Ltd', 'FinTech Global', 'FinTech Global', 'FinTech Global'],
    'Environmental_Score': [75, 78, 80, 80, 82, 85, 70, 72, 75, 68, 70, 73],
    'Social_Score': [80, 82, 85, 85, 88, 90, 75, 78, 80, 70, 72, 75],
    'Governance_Score': [70, 72, 75, 75, 78, 80, 65, 68, 70, 60, 65, 68],
    'Overall_ESG_Score': [75, 77, 80, 80, 82, 85, 70, 73, 75, 66, 69, 72],
    'CO2_Emissions (tons)': [5000, 4800, 4600, 4000, 3800, 3600, 5500, 5300, 5100, 6000, 5800, 5600],
    'Diversity_Initiatives': ['Employee Diversity', 'Gender Equality', 'Inclusive Culture', 'Sustainable Sourcing', 'Community Engagement', 'Green Initiatives',
                              'Health & Safety', 'Employee Well-being', 'Community Support', 'Ethical Practices', 'Governance Training', 'Transparent Reporting'],
    'Governance_Transparency': ['High', 'High', 'Very High', 'High', 'High', 'Very High', 'Medium', 'High', 'High', 'Medium', 'High', 'High']
})

# Step 2: Construct the MultiDiGraph

# Create a multi-directed graph
G = nx.MultiDiGraph()

# Add nodes for each company
companies = set(b2b_sales_data['Seller_Company']).union(set(b2b_sales_data['Buyer_Company']))
G.add_nodes_from(companies)

# Add edges based on B2B sales transactions
for _, row in b2b_sales_data.iterrows():
    G.add_edge(row['Seller_Company'], row['Buyer_Company'], key='Sales', weight=row['Sales_Amount'])

# Add edges based on ESG scores (weighted by Overall_ESG_Score)
for _, row in esg_data.iterrows():
    G.add_edge(row['Company'], row['Company'], key='ESG', weight=row['Overall_ESG_Score'])

# Add edges based on financial relationships (e.g., Net_Income)
for _, row in financial_data.iterrows():
    G.add_edge(row['Company'], row['Company'], key='Financial', weight=row['Net_Income'])

# Function to predict potential customers
def predict_potential_customers(company, G):
    neighbors = set(G.neighbors(company))
    potential_customers = set(G.nodes) - neighbors - {company}
    return list(potential_customers)

# Function to get metadata and financial chart of a company
def get_company_metadata(company):
    if company not in financial_data['Company'].values:
        return None

    financials = financial_data[financial_data['Company'] == company]
    esg = esg_data[esg_data['Company'] == company]
    
    if financials.empty or esg.empty:
        return None

    esg = esg.iloc[0].to_dict()

    # Generate financial chart
    fig, ax = plt.subplots()
    ax.plot(financials['Year'], financials['Revenue'], label='Revenue')
    ax.plot(financials['Year'], financials['Net_Income'], label='Net Income')
    ax.set_title(f'Financials of {company}')
    ax.set_xlabel('Year')
    ax.set_ylabel('Amount')
    ax.legend()
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)

    metadata = {
        'company': company,
        'financials': financials.to_dict(orient='records'),
        'esg': esg,
        'chart': image_base64
    }

    return metadata

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    company = request.form['company']
    if company not in G.nodes:
        return jsonify({'error': 'Company not found'}), 404

    current_customers = list(G.neighbors(company))
    potential_customers = predict_potential_customers(company, G)
    return jsonify({
        'current_customers': current_customers,
        'potential_customers': potential_customers
    })

@app.route('/metadata/<company>')
def metadata(company):
    metadata = get_company_metadata(company)
    if metadata is None:
        return jsonify({'error': 'Company data not found'}), 404
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(debug=True)
