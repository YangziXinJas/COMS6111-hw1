import pprint

from googleapiclient.discovery import build


def main():
    # Build a service object for interacting with the API. Visit
    # the Google APIs Console <http://code.google.com/apis/console>
    # to get an API key for your own application.
    service = build(
        "customsearch", "v1", developerKey="AIzaSyDJZCCaWHDkdmn1dENt6Pynp9mcykVHtpg"
    )

    res = (
        service.cse()
        .list(
            q=input("Enter search words: "),
            cx="c6f4622f0b9a14652",
        )
        .execute()
    )
    precision = input("Enter a value between 0 and 1: ")

    result = res['items']


    relevance = -1
    while relevance / 10 < float(precision) or relevance == 0:
        relevance = 0
        for item in result:
            d = {}
            d['title'] = item['title']
            d['url'] = item['link']
            d['description'] = item['snippet']

            pprint.pprint(d)
            if input("Relevant (Y/N)? ").capitalize() == 'Y':
                relevance += 1
        print("\n\n\n\n\n")
        pprint.pprint(relevance/10)
    


        

    #pprint.pprint(res)


if __name__ == "__main__":
    main()