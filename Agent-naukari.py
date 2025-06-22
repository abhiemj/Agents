import getpass
import os
from langchain_core.tools import tool
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, Tool
from dotenv import load_dotenv
load_dotenv()  # This loads the .env file into os.environ


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


@tool("naukri_job_applier")
def apply_jobs_on_naukri(job_title: str) -> str:
    """
    Logs into Naukri.com and applies to the first few jobs matching the job title.
    Uses Naukri email and password from environment variables.
    Input: job_title (string)
    Output: List of job titles and URLs where applications were attempted.
    """
    import json
    import os
    email = os.environ.get("NAUKRI_EMAIL")
    print(email)
    password = os.environ.get("NAUKRI_PASS")
    if not email or not password:
        return "❌ Naukri credentials not set in environment variables. Please set NAUKRI_EMAIL and NAUKRI_PASS."
    try:
        # Launch browser
        driver = webdriver.Chrome()
        driver.get("https://www.naukri.com/")
        time.sleep(3)
        driver.find_element(By.ID, "login_Layer").click()
        time.sleep(2)
        username_field = driver.find_element(By.XPATH, '//input[@placeholder="Enter your active Email ID / Username"]')
        username_field.send_keys(email)
        password_field = driver.find_element(By.XPATH, '//input[@placeholder="Enter your password"]')
        password_field.send_keys(password)
        password_field.send_keys(Keys.RETURN)
        time.sleep(3)
        search_placeholder = driver.find_element(By.CLASS_NAME, "nI-gNb-sb__placeholder")
        search_placeholder.click()
        time.sleep(1)
        active_input = driver.find_element(By.XPATH, '//input[@placeholder="Enter keyword / designation / companies"]')
        active_input.send_keys(job_title)
        time.sleep(1) 

        search_icon = driver.find_element(By.CLASS_NAME, "ni-gnb-icn-search")
        search_icon.click()
        time.sleep(5)
        job_listings = driver.find_elements(By.CLASS_NAME, "cust-job-tuple")
        applied = []
        for job in job_listings[:5]:
            title_element = job.find_element(By.CLASS_NAME, "title")
            job_title_text = title_element.text
            print("Applying to:", title_element.text)
            title_element.click()
            time.sleep(3)
            driver.switch_to.window(driver.window_handles[-1])
            time.sleep(3)
            try:
                apply_button = driver.find_element(By.ID, "apply-button")
                apply_button.click()
                # --- Chatbot popup handling ---
                time.sleep(2)
                from selenium.common.exceptions import NoSuchElementException
                chatbot_success = False
                while True:
                    try:
                        drawer = driver.find_element(By.CSS_SELECTOR, ".chatbot_DrawerContentWrapper")
                        question_elems = drawer.find_elements(By.CSS_SELECTOR, ".botMsg.msg span")
                        if question_elems and len(question_elems) >= 2:
                            greeting_text = question_elems[-2].text.strip()
                            question_text = question_elems[-1].text.strip()
                            print("Chatbot message:", greeting_text)
                            print("Chatbot question:", question_text)
                        elif question_elems:
                            question_text = question_elems[-1].text.strip()
                            print("Chatbot question:", question_text)
                        else:
                            # No more questions, assume success
                            chatbot_success = True
                            break
                        chip_elems = drawer.find_elements(By.CSS_SELECTOR, ".chatbot_Chip.chipInRow.chipItem span")
                        chip_clicked = False
                        if chip_elems:
                            options = [chip.text.strip() for chip in chip_elems]
                            print("Options:", ", ".join(options))
                            user_answer = ""
                            while user_answer.lower() not in [opt.lower() for opt in options]:
                                user_answer = input(f"Please select one of the options ({', '.join(options)}): ").strip()
                            for chip in chip_elems:
                                if chip.text.strip().lower() == user_answer.lower():
                                    chip.click()
                                    chip_clicked = True
                                    time.sleep(1)
                                    break
                        else:
                            user_answer = input("Please answer the recruiter's question: ")
                            try:
                                input_box = drawer.find_element(By.CSS_SELECTOR, ".textAreaWrapper .textArea[contenteditable='true']")
                                input_box.click()
                                input_box.send_keys(user_answer)
                                input_box.send_keys(Keys.RETURN)
                                time.sleep(1)
                            except NoSuchElementException:
                                print("No valid input method found for the answer.")
                        # After answering, click the Save button if present
                        try:
                            save_btn = drawer.find_element(By.CSS_SELECTOR, ".sendMsg.sendMsg")
                            if save_btn.is_displayed() and save_btn.is_enabled():
                                save_btn.click()
                                time.sleep(2)
                        except NoSuchElementException:
                            pass
                        time.sleep(2)
                    except NoSuchElementException:
                        # Drawer not found, assume success
                        chatbot_success = True
                        break
                if chatbot_success:
                    applied.append({
                        "title": job_title_text,
                        "url": driver.current_url,
                        "status": "✅ Applied"
                    })
                else:
                    applied.append({
                        "title": job_title_text,
                        "url": driver.current_url,
                        "status": "❌ Application may not have completed"
                    })
                # --- End chatbot popup handling ---
            except:
                applied.append({
                    "title": job_title_text,
                    "url": driver.current_url,
                    "status": "❌ Apply button not found"
                })
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(2)
        driver.quit()
        return json.dumps(applied, indent=2)
    except Exception as e:
        driver.quit()
        return f" Job application failed: {str(e)}"
    
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [apply_jobs_on_naukri]  # Define your tools here

# 3. Bind memory to the agent
agent = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": (
            "You are a helpful assistant that helps users apply for jobs on Naukri.com. "
            "Whenever the user wants to apply for a job, always ask for the job title if not already provided. "
            "Never assume or make up job titles. Only call the tool when you have the job title. "
            "Naukri credentials are set as environment variables."
        )
    }
)
    
while True:
    try:
        user_input = input("You: ")
        result = agent.invoke({"input": user_input})
        print("Agent:", result["output"] if isinstance(result, dict) and "output" in result else result)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")

