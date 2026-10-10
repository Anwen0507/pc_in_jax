def init_activities_with_ffwd(model, input_):
    # feedforward init for a SINGLE example:
    # activities[l] = layer_l(activities[l - 1]), starting from input_
    activities = [model[0](input_)]
    for l in range(1, len(model)):
        activities.append(model[l](activities[l - 1]))
    return activities