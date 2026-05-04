from preprocessing import preprocess_text

# testing to read doc01.txt 
try:
    with open("documents/doc01.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()
        
    # since it's too long, lets test the first 500 char only
    sample_content = raw_text[:500]
    clean_content = preprocess_text(sample_content)
    
    print("--- Original (First 500 chars) ---")
    print(sample_content)
    print("\n--- Processed Tokens ---")
    print(clean_content)

except FileNotFoundError:
    print("Hata: 1.txt dosyası documents klasöründe bulunamadı!")
