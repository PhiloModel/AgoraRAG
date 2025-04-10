from langchain.document_loaders import PyPDFLoader
import glob


# Pobieranie zestawu pdfów z wybranego folderu
def load_pdfs_from_dir(dir_path):
    # Wybieranie wszystkich plików folderu które sa pdf'ami
    pdf_files = glob.glob(dir_path + "*.pdf")

    all_pages = []
    
    for pdf_path in pdf_files:
        print(f'Processing {pdf_path}')
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        all_pages.extend(pages)
    
    print(f"Loaded {len(all_pages)} docs.")

    return all_pages