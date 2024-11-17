from src import model

testcases = [
    ("What is the benefit of the mobile app for Move 4 Life?", ["Move 4 Life", "refresher courses", "convenience", "grow engagement"]),
    ("What concerts were planned for O2's 10th birthday?", ["Ed Sheeran", "Alt-J", "Jamiroquai", "Foo Fighters"]),
    ("Did EE work with McDonalds?", ["I don't know"]),
]

def test_model():
    for question, expected_keywords in testcases:
        answer = model.answer_question(question)
        for keyword in expected_keywords:
            assert keyword in answer, f"Expected keyword '{keyword}' not found in answer"