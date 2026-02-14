from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool


@CrewBase
class MyFinResearcher:
    """MyFinResearcher crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"], verbose=True, tools=[SerperDevTool()]  # type: ignore[index]
        )

    @agent
    def analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["analyst"], verbose=True  # type: ignore[index]
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config["analysis_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyFinResearcher crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
