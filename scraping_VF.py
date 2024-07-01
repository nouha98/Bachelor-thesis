import requests
from recipe_scrapers import scrape_html


url = ["https://www.allrecipes.com/recipe/158968/spinach-and-feta-turkey-burgers/","https://www.jamieoliver.com/recipes/duck-recipes/roast-duck-with-marsala-gravy/" ]
url.append("https://www.jamieoliver.com/recipes/pasta-recipes/beautiful-courgette-penne-carbonara/")

def R_scrape(url):

	html = requests.get(url).content

	scraper = scrape_html(html=html, org_url=url)


	#print("title",scraper.title())
	#scraper.total_time()
	#scraper.yields()
	recipe_name = scraper.title()
	ingredients = scraper.ingredients()
	instructions = scraper.instructions()
	#print("ingredients: ",scraper.ingredients())
	#print("instructions: ",scraper.instructions())  # or alternatively for results as a Python list: scraper.instructions_list()
	img_URL = scraper.image()

	

	#scraper.host()
	#scraper.links()

	#print("Nutrients: ",scraper.nutrients())
	Nutrients = scraper.nutrients()
	return recipe_name, ingredients, instructions, img_URL

#recipe_name, ingredients, instructions, img_URL = R_scrape('https://www.allrecipes.com/recipe/158968/spinach-and-feta-turkey-burgers/')
#print(ingredients)
#print(instructions)
#print(img_URL)

