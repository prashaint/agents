from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel
from pydantic.fields import Field


class TrendingCompany(BaseModel):
    """Trending company"""

    name: str = Field(description="Name of the company")
    ticker: str = Field(description="Ticker of the company")
    reason: str = Field(description="Reason for the company being trending in the news")


class TrendingCompanyList(BaseModel):
    """List of trending companies"""

    companies: List[TrendingCompany] = Field(
        description="List of trending companies in the news"
    )


class TrendingCompanyReport(BaseModel):
    """Trending company report"""

    report: str = Field(description="Report on the trending company")


@CrewBase
class MyStockAnalyzer:
    """MyStockAnalyzer crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"], verbose=True  # type: ignore[index]
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config["reporting_analyst"],  # type: ignore[index]
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],  # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config["reporting_task"],  # type: ignore[index]
            output_file="report.md",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the MyStockAnalyzer crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
