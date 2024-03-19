
class Article:
    def __init__(self, title, authors, year, nr_of_pages):
        self.title = title
        self.authors = authors  # list
        self.year = year
        self.nr_of_pages = nr_of_pages

    def add_author(self, author_name):
        self.authors.append(author_name)
        print("Another author is: {}".format(author_name))

    def print_authors(self):
        for i in self.authors:
            print(i, end=", ")
        print()

    def change_title(self, new_title):
        self.title = new_title
        print("Title changed to {}".format(self.title))

    def get_info(self):
        print("Title: {}".format(self.title))
        print("Authors: ", end="")
        self.print_authors()
        print("Year: {}".format(self.year))
        print("Number of pages: {}".format(self.nr_of_pages))

    def add_page(self):
        self.nr_of_pages += 1
        print("Number of pages is {} now".format(self.nr_of_pages))


article = Article("Nudny artykul", ["Jan Kowalski", "Maciej Nowak"], 2024, 25)
article.get_info()
print()

article.change_title("Fajny artykul")
article.add_page()
article.add_author("Jerzy")
print()

article.get_info()
