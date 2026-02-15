from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel
from pydantic.fields import Field
from crewai_tools import SerperDevTool

# from push_tool import PushNotificationTool
from crewai.memory import ShortTermMemory, LongTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


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


class TrendingCompanyResearch(BaseModel):
    """Detailed research on trending company"""

    name: str = Field(description="Name of the company")
    market_position: str = Field(description="Current market position of the company")
    future_outlook: str = Field(description="Future outlook of the company")
    investment_potential: str = Field(
        description="Investment potential and suitability for investment in the company"
    )


class TrendingCompanyResearchList(BaseModel):
    """List of trending companies research"""

    companies: List[TrendingCompanyResearch] = Field(
        description="Comprehensive research on all trending companies"
    )


@CrewBase
class MyStockAnalyzer:
    """MyStockAnalyzer crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(
            config=self.agents_config["trending_company_finder"], verbose=True, tools=[SerperDevTool()], memory=True  # type: ignore[index]
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["financial_researcher"],  # type: ignore[index]
            verbose=True,
            tools=[SerperDevTool()],
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config["stock_picker"],  # type: ignore[index]
            verbose=True,
            # tools=[SerperDevTool(), PushNotificationTool()],
            tools=[SerperDevTool()],
            memory=True,
        )

    @task
    def find_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["find_trending_companies"],
            output_pydantic_model=TrendingCompanyList,  # type: ignore[index]
        )

    @task
    def research_trending_companies(self) -> Task:
        return Task(
            config=self.tasks_config["research_trending_companies"],
            output_pydantic_model=TrendingCompanyResearchList,  # type: ignore[index]
        )

    @task
    def pick_best_company(self) -> Task:
        return Task(config=self.tasks_config["pick_best_company"])

    @crew
    def crew(self) -> Crew:
        """Creates the MyStockAnalyzer crew"""

        manager = Agent(
            config=self.agents_config["manager"],  # type: ignore[index]
            allow_delegation=True,
            verbose=True,
        )

        short_term_memory = ShortTermMemory(
            storage=RAGStorage(
                type="short_term",
                path="./memory",
                embedder_config={
                    "provider": "openai",
                    "config": {
                        "model_name": "text-embedding-3-small",
                    },
                },
            )
        )
        long_term_memory = LongTermMemory(
            storage=LTMSQLiteStorage(db_path="./memory/long_term.db")
        )
        entity_memory = EntityMemory(
            storage=RAGStorage(
                type="short_term",
                path="./memory",
                embedder_config={
                    "provider": "openai",
                    "config": {
                        "model_name": "text-embedding-3-small",
                    },
                },
            )
        )
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            tracing=True,
            memory=True,
            short_term_memory=short_term_memory,
            long_term_memory=long_term_memory,
            entity_memory=entity_memory,
        )
