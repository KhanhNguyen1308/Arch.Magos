from googlesearch import search
search_results = search("This is my query", 1)
for result in search_results:
    print(result)