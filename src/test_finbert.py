from finbert_sentiment import FinBertSentiment

def main():
    clf = FinBertSentiment()

    samples = [
        "Apple shares rise after strong earnings and upbeat guidance.",
        "The company warned investors about declining revenue and higher costs.",
        "Markets were mixed today as traders awaited the Fed decision.",
        "Tesla faces lawsuit over alleged misleading statements.",
        "Gold prices surge amid global uncertainty.",
    ]

    df = clf.predict_df(samples, batch_size=8)
    print(df.to_string(index=False))

    # Optionnel: export CSV
    df.to_csv("finbert_test_output.csv", index=False)
    print("\nSaved: finbert_test_output.csv")

if __name__ == "__main__":
    main()