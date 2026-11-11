"""Strict-enough structured CV models matching the 20 supplied synthetic documents."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    start_year: int = Field(ge=1950, le=2100)
    end_year: int = Field(ge=1950, le=2100)

    @model_validator(mode="after")
    def check_years(self) -> "Job":
        if self.end_year < self.start_year:
            raise ValueError("end_year must be >= start_year")
        return self


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree: str = Field(min_length=1)
    graduation_year: int = Field(ge=1950, le=2100)


class CVData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(min_length=1)
    email: str = Field(min_length=3, pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
    phone: str = Field(min_length=5)
    location: str = Field(min_length=1)
    years_experience: int = Field(ge=0, le=80)
    skills: list[str] = Field(min_length=1)
    work_experience: list[Job] = Field(min_length=1)
    education: Education

    @field_validator("skills")
    @classmethod
    def check_skills(cls, values: list[str]) -> list[str]:
        if any(not s.strip() for s in values):
            raise ValueError("skill names cannot be empty")
        return values
