# LLM post generator
An app that uses the SERP API together with the gpt-3.5-turbo model to generate a Reddit post. This is achieved by using
LangChain and prompt templates to chain everything together, and generate the desired result.

The following process underpins the end-to-end app functionality:
* A topic of choice is entered into the web application which forms the [SERP](https://serper.dev/) query
* The next step is used to find the best URL using a prompt template and the gpt-3.5-turbo model
* Once the URL are returned, LangChain is used to fetch the data from the URL
* Then the process summarises the fetched data, using the gpt-3.5-turbo and prompt template
* Finally, LangChain and prompt templates are used in tandem with gpt-3.5-turbo to generate a Reddit post


## System design

![system-design](img/system-design.drawio.png)


## Models

- [gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)


## Requirements

```bash
python -m pip install -r requirements.txt
```


## Run App Locally

```bash
source build.sh
```


## Run App with Streamlit Cloud

[Launch App](https://llm-post-generator.streamlit.app/)


## License

Distributed under the MIT License. See `LICENSE` for more information.
