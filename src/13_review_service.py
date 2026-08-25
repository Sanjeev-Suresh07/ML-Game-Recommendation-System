import requests
import certifi


# ============================================================
# GET STEAM REVIEWS
# ============================================================

def get_game_reviews(app_id, num_reviews=5):

    url = (
        f"https://store.steampowered.com/appreviews/"
        f"{app_id}?json=1"
    )

    params = {
        "filter": "all",
        "language": "english",
        "num_per_page": num_reviews,
        "purchase_type": "all"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10,
            verify=certifi.where()
        )

        response.raise_for_status()

        data = response.json()

    except Exception as e:

        print(
            "Error getting reviews:",
            e
        )

        return None


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    if data.get("success") != 1:

        print(
            "Could not get review data."
        )

        return None


    # ========================================================
    # REVIEW SUMMARY
    # ========================================================

    summary = data.get(
        "query_summary",
        {}
    )

    result = {

        "review_score": summary.get(
            "review_score",
            0
        ),

        "review_score_description": summary.get(
            "review_score_desc",
            "No rating"
        ),

        "total_reviews": summary.get(
            "total_reviews",
            0
        ),

        "positive_reviews": summary.get(
            "total_positive",
            0
        ),

        "negative_reviews": summary.get(
            "total_negative",
            0
        ),

        "reviews": []
    }


    # ========================================================
    # INDIVIDUAL REVIEWS
    # ========================================================

    for review in data.get(
        "reviews",
        []
    ):

        result["reviews"].append({

            "review": review.get(
                "review",
                ""
            ),

            "recommended": review.get(
                "voted_up",
                False
            ),

            "helpful_votes": review.get(
                "votes_up",
                0
            )
        })


    return result


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "========================================"
    )

    print(
        "STEAM REVIEW TEST"
    )

    print(
        "========================================"
    )


    # Counter-Strike 2 Steam AppID
    app_id = 730


    result = get_game_reviews(
        app_id,
        num_reviews=5
    )


    if result is not None:

        print(
            "\nReview Score:",
            result["review_score"]
        )

        print(
            "Rating:",
            result[
                "review_score_description"
            ]
        )

        print(
            "Total Reviews:",
            result["total_reviews"]
        )

        print(
            "Positive Reviews:",
            result["positive_reviews"]
        )

        print(
            "Negative Reviews:",
            result["negative_reviews"]
        )


        print(
            "\nSample Reviews:"
        )


        for number, review in enumerate(
            result["reviews"],
            start=1
        ):

            print(
                f"\n{number}. "
                f"{review['review'][:200]}"
            )

            print(
                "Recommended:",
                review["recommended"]
            )

            print(
                "Helpful votes:",
                review["helpful_votes"]
            )


    print(
        "\n========================================"
    )

    print(
        "REVIEW TEST COMPLETE"
    )

    print(
        "========================================"
    )