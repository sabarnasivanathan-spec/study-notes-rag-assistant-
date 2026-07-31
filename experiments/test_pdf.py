from pypdf import PdfReader

reader = PdfReader("B.Tech. IT.pdf")  # replace with your actual filename

text = ""
for page in reader.pages:
    text += page.extract_text()

print(text[:1000])  # just print the first 1000 characters to check it worked
print(f"\n\nTotal characters extracted: {len(text)}")