import requests
import argparse


def get_user_auth_session_cookie(url, username, password):
    get_response = requests.get(url)

    if "auth" in get_response.url:
        credentials = {"login": username, "password": password}

        # Authenticate user
        session = requests.Session()
        print(get_response.url)
        session.post(get_response.url, data=credentials)
        print(session.cookies)
        cookie_auth_key = "authservice_session"
        cookie_auth_value = session.cookies.get(cookie_auth_key)

        if cookie_auth_value:
            return cookie_auth_key + "=" + cookie_auth_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", type=str, help="URL to kubeflow server")
    parser.add_argument("--username", type=str, help='username to login kubeflow dex')
    parser.add_argument("--password", type=str, help='password to login kubeflow dex')
    args = parser.parse_args()

    endpoint = args.endpoint
    api_username = args.username
    api_password = args.password
    api_endpoint = f"{endpoint}/pipeline"

    session_cookie = get_user_auth_session_cookie(endpoint, api_username, api_password)
    with open("session_cookie.txt", 'w') as f:
        f.write(session_cookie)