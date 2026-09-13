from __future__ import annotations

from dataclasses import dataclass

from generation.catalog import OperatorParameter


@dataclass(frozen=True, slots=True)
class BoundArgument:
    parameter: OperatorParameter
    parameter_index: int
    argument_index: int


@dataclass(frozen=True, slots=True)
class ArgumentBindingIssue:
    code: str
    argument_index: int | None
    parameter_name: str | None


@dataclass(frozen=True, slots=True)
class ArgumentBinding:
    assignments: tuple[BoundArgument, ...]
    issues: tuple[ArgumentBindingIssue, ...]


def bind_operator_arguments(
    parameters: tuple[OperatorParameter, ...],
    argument_names: tuple[str | None, ...],
) -> ArgumentBinding:
    parameters_by_name = {
        parameter.name: (index, parameter)
        for index, parameter in enumerate(parameters)
    }
    assigned_counts = [0 for _ in parameters]
    positional_index = 0
    assignments: list[BoundArgument] = []
    issues: list[ArgumentBindingIssue] = []

    for argument_index, argument_name in enumerate(argument_names):
        if argument_name is not None:
            matched = parameters_by_name.get(argument_name)
            if matched is None:
                issues.append(
                    ArgumentBindingIssue(
                        code="unknown_named_argument",
                        argument_index=argument_index,
                        parameter_name=argument_name,
                    )
                )
                continue
            parameter_index, parameter = matched
            if assigned_counts[parameter_index] and not parameter.variadic:
                issues.append(
                    ArgumentBindingIssue(
                        code="duplicate_argument",
                        argument_index=argument_index,
                        parameter_name=parameter.name,
                    )
                )
                continue
        else:
            while positional_index < len(parameters):
                candidate = parameters[positional_index]
                if candidate.variadic:
                    parameter_index = positional_index
                    parameter = candidate
                    break
                if assigned_counts[positional_index] == 0:
                    parameter_index = positional_index
                    parameter = candidate
                    positional_index += 1
                    break
                positional_index += 1
            else:
                issues.append(
                    ArgumentBindingIssue(
                        code="too_many_positional_arguments",
                        argument_index=argument_index,
                        parameter_name=None,
                    )
                )
                continue

        assigned_counts[parameter_index] += 1
        assignments.append(
            BoundArgument(
                parameter=parameter,
                parameter_index=parameter_index,
                argument_index=argument_index,
            )
        )

    for parameter_index, parameter in enumerate(parameters):
        if not parameter.optional and assigned_counts[parameter_index] == 0:
            issues.append(
                ArgumentBindingIssue(
                    code="missing_argument",
                    argument_index=None,
                    parameter_name=parameter.name,
                )
            )
    return ArgumentBinding(
        assignments=tuple(assignments),
        issues=tuple(issues),
    )
