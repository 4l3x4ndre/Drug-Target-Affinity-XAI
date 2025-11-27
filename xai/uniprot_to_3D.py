import requests
import py3Dmol

def get_swissmodel_models(uniprot_id):
    # Endpoint for searching models by UniProt accession
    url = f"https://swissmodel.expasy.org/repository/uniprot/{uniprot_id}.json"
    
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    
    if 'result' in data and 'structures' in data['result']:
        return data['result']['structures'][0]

def download_model_coordinates(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        f.write(response.content)


def uniprot_to_html(uniprot_id):
    """From a uniprof_id, return the html content"""
    models = get_swissmodel_models(uniprot_id)

    if models:
        print(f"Model details: {models}, Quality: {models.get('coverage', 'N/A')}")

        # Download coordinates (usually PDB format file url)
        coord_url = models['coordinates']

        response = requests.get(coord_url)
        response.raise_for_status()

        pdb_data = response.content.decode('utf-8')

        view = py3Dmol.view(width=800, height=600)
        view.addModel(pdb_data, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        html_content = view._make_html()

        return html_content
    else:
        print(f"No models found for UniProt ID {uniprot_id}")
        return None
