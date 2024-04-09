from flask import Flask, render_template, request
from course_recommender import recommend_courses_by_skill

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/courses")
def courses():
    return render_template("courses.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/result", methods=['GET'])
def result():
    skill = request.args.get('skill')
    recommended_courses = recommend_courses_by_skill(skill)
    processed_courses = [convert_to_camel_case(course) for course in recommended_courses]
    return render_template("result.html", courses=processed_courses)

def convert_to_camel_case(course_name):
    words = course_name.split()
    camel_case_words = [word.capitalize() for word in words]
    
    # Insert spaces before capital letters
    camel_case_with_spaces = ''
    for word in camel_case_words:
        for char in word:
            if char.isupper():
                camel_case_with_spaces += ' ' + char
            else:
                camel_case_with_spaces += char
    return camel_case_with_spaces.lstrip()  # Remove leading space if any

if __name__ == "__main__":
    app.run(debug=True)