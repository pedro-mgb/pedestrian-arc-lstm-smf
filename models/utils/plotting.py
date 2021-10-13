"""
Created on 21/08/2021.
Utilities for plotting trajectories and predictions
"""
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors
from matplotlib.patches import Arc as PltArc

from models.interaction_modules.shape_based import PoolingShape
from models.utils.evaluator import pred_data_call, TrajectoryType
from models.utils.utils import center_scene, normalize_sequences, relative_traj_to_abs

alpha_marker = 0.7  # fixed value for transparency
neigh_color = 'dimgray'
pool_shape_color = 'black'

anim_list = []

class AnimatedSituation:
    """
    Class to create and store a FuncAnimation of a situation that may contain social interactions.
    Assumes Trajnet++ format, with a primary pedestrian and neighbours.
    Since this relies on matplotlib, make sure you maintain a reference to this object till the end of the program
    running, or until plt.show() is called. Otherwise, the animation may fail to be shown and saved.
    """

    num_animations = 0

    def __init__(self, args, obs_traj, gt, predictions, environment, model_list, model_labels, x_limits=None,
                 y_limits=None):
        """
        Create a situation to animate. Able to configure several aspects of the animation and plot predictions from
        different models in the same figure.
        This animation creation will cover both the past and future trajectory.
        :param args: command line arguments containing several options to configure the animation
        :param obs_traj: tensor of shape [obs_len, num_peds, 2]. Observed trajectories of all pedestrians in a situation
        :param gt: tensor of shape [pred_len, num_peds, 2]. Ground truth trajectories of all pedestrians in a situation
        :param predictions: tensor of shape [num_models, pred_len, num_peds, 2]. Predicted trajectories of all
        pedestrians in a situation, for num_models supplied (num_models >= 1)
        :param environment: The scene-specific environment to plot, or None if there is no environment to plot
        :param model_labels: list of length num_models, containing labels to identify each of the models
        :param model_list: list of length num_models, containing the instances of each model used; may contain useful
        information about particular models, like the size of FOV shape pooling being used.
        :param x_limits: limits of the plot along x axis, or None if no limits are meant to be used
        :param y_limits: limits of the plot along y axis, or None if no limits are meant to be used
        """
        self.args = args
        self.obs, self.gt, self.pred = obs_traj, gt, predictions
        self.num_peds = gt.shape[1]
        self.labels, self.num_models = model_labels, len(model_labels)
        self.environment = environment
        self.x_limits, self.y_limits = x_limits, y_limits

        models_with_interactions = ['interaction' in model.__class__.__name__.lower() for model in model_list]

        fig, ax = plt.subplots()

        # list of objects to plot
        self.objects_plt = []
        self.objects_obs = []
        self.objects_gt = []
        self.objects_pred = []  # list of lists
        plt_prediction_colours = []
        self.model_pooling_shapes, pool_shape_selected = [], False

        # plot the lines for the past trajectories, and initialize the ones for GT and model predictions
        for i in range(self.num_peds):
            lns_pred = []
            if i == 0:
                # primary pedestrian highlighted (assuming Trajnet++)
                ln, = plt.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], linewidth=3, marker='o', markersize=5, label='OBS')
                ln_gt, = plt.plot([], [], linewidth=3, marker='o', markersize=5, label='GT', markevery=(2, 1))
                ln_color, ln_gt_color = ln.get_color(), ln_gt.get_color()
                ln.set_markerfacecolor(matplotlib.colors.to_rgba(ln_color, alpha_marker))
                ln.set_markeredgecolor(matplotlib.colors.to_rgba(ln_color, alpha_marker))
                ln_gt.set_markerfacecolor(matplotlib.colors.to_rgba(ln_gt_color, alpha_marker))
                ln_gt.set_markeredgecolor(matplotlib.colors.to_rgba(ln_gt_color, alpha_marker))
                for (idx, label) in enumerate(model_labels):
                    ln_pred, = plt.plot([], [], linewidth=3, marker='o', markersize=5, label=label)
                    ln_pred_color = ln_pred.get_color()
                    plt_prediction_colours.append(ln_pred_color)
                    ln_pred.set_markerfacecolor(matplotlib.colors.to_rgba(ln_pred_color, alpha_marker))
                    ln_pred.set_markeredgecolor(matplotlib.colors.to_rgba(ln_pred_color, alpha_marker))
                    lns_pred.append(ln_pred)
                    if args.plot_pool_shape and models_with_interactions[idx] and not pool_shape_selected:
                        pool_shape_selected = True  # only one model with pooling shape to be drawn
                        social_args = model_list[idx].interaction_module.args
                        if social_args.shape == PoolingShape.GRID:
                            raise Exception('Not implemented yed')
                        else:
                            # arc shape
                            arc_radius, arc_angle = social_args.arc_radius, social_args.arc_angle
                            arc_position, rel_pos = obs_traj[-1, i], obs_traj[-1, i] - obs_traj[-2, i]
                            arc_direction = torch.atan2(rel_pos[1], rel_pos[0])
                            drawn_arc = ArcPatch(arc_radius, arc_angle, arc_position, arc_direction, ax)
                            self.model_pooling_shapes.append(drawn_arc)
                    else:
                        self.model_pooling_shapes.append(None)

            else:
                # neighbour (assuming Trajnet++)
                neigh_m_color = matplotlib.colors.to_rgba(neigh_color, alpha_marker)
                if self.args.anim_neighbour_motion:
                    # ln will contain current neighbour position, ln_gt will contain arrow with neighbour motion
                    # ln_pred contains predicted neighbour motion
                    ln, = plt.plot(obs_traj[-1, i, 0], obs_traj[-1, i, 1], linewidth=2, color=neigh_color, marker='o',
                                   markersize=3, markeredgecolor=neigh_m_color, markerfacecolor=neigh_m_color,
                                   label='NEIGH' if i == 1 else None)
                    motion = obs_traj[-1, i] - obs_traj[-2, i]
                    if torch.all(motion == 0):
                        # plot single point indicating neighbour is stationary
                        ln_gt = plt.scatter(obs_traj[-1, i, 0], obs_traj[-1, i, 1], s=20, color=neigh_color)
                    else:
                        # plot arrow with motion
                        ln_gt = plt.arrow(obs_traj[-2, i, 0], obs_traj[-2, i, 1], motion[0],
                                          motion[1], color=neigh_color, width=0.05, head_width=0.3)
                    if not self.args.ignore_neighbour_pred:
                        for (idx, label) in enumerate(model_labels):
                            ln_pred_color = plt_prediction_colours[idx]
                            ln_pred = plt.plot(float('nan'), float('nan'), float('nan'), float('nan'),
                                               color=ln_pred_color)
                            lns_pred.append(ln_pred)
                else:
                    ln, = plt.plot(obs_traj[:, i, 0], obs_traj[:, i, 1], linewidth=2, color=neigh_color, marker='o',
                                   markersize=3, markeredgecolor=neigh_m_color, markerfacecolor=neigh_m_color,
                                   label='NEIGH' if i == 1 else None)
                    ln_gt, = plt.plot([], [], linewidth=2, color=neigh_color, marker='o', markersize=3)
                    if not self.args.ignore_neighbour_pred:
                        for (idx, label) in enumerate(model_labels):
                            ln_pred_color = plt_prediction_colours[idx]
                            ln_pred_marker_color = matplotlib.colors.to_rgba(plt_prediction_colours[idx], alpha_marker)
                            ln_pred, = plt.plot([], [], linewidth=2, color=ln_pred_color, marker='o', markersize=3,
                                                markeredgecolor=ln_pred_marker_color,
                                                markerfacecolor=ln_pred_marker_color)
                            lns_pred.append(ln_pred)
            self.objects_obs.append(ln)
            self.objects_gt.append(ln_gt)
            self.objects_pred.append(lns_pred)

        num_frames = gt.shape[0] - 1

        # configure title for the plot animation
        self.plot_title_base = ''
        self.plot_title = ''
        if self.args.plot_title is not None and self.args.plot_title:
            self.plot_title_base = self.args.plot_title
            self.plot_title = self.plot_title_base
        if self.args.timestamp_in_plot_title:
            self.plot_title = self.plot_title_base + (f' (t={obs_traj.shape[0]})' if self.plot_title_base
                                                      else f'Time t={obs_traj.shape[0]}')
        if self.plot_title:
            plt.title(self.plot_title, fontdict={'fontsize': 20})

        if self.environment:
            self.environment.plot(plt)

        # initialize some peculiar points of this animation
        def __init_anim__():
            for s in self.model_pooling_shapes:
                if s is not None and isinstance(s, ArcPatch):
                    return s.arc  # return the modified patch

        # update the animation at instant #frame_id
        def __update__(frame_id):
            if frame_id == 0:
                return  # the first frame of the animation to contain only the observed trajectory
            shape_updated = False
            frame = num_frames if frame_id > num_frames else frame_id
            gt_frame, pred_frame = self.gt[:(frame + 1)], self.pred[:, :(frame + 1)]
            for j in range(self.num_peds):
                if j > 0 and self.args.anim_neighbour_motion:
                    _motion = self.gt[frame, j] - self.gt[frame - 1, j]
                    self.objects_obs[j].set_data(self.gt[frame, j, 0], self.gt[frame, j, 1])
                    self.objects_gt[j].remove()
                    if torch.all(_motion == 0):
                        self.objects_gt[j] = plt.scatter(self.gt[frame, j, 0], self.gt[frame, j, 1], s=20,
                                                         color=neigh_color)
                    else:
                        self.objects_gt[j] = plt.arrow(self.gt[frame - 1, j, 0], self.gt[frame - 1, j, 1], _motion[0],
                                                       _motion[1], color=neigh_color, width=0.05, head_width=0.3)
                    '''
                    self.objects_gt[j].x, self.objects_gt[j].y = self.gt[frame, j, 0], self.gt[frame, j, 1]
                    self.objects_gt[j].dx, self.objects_gt[j].dy = _motion[0], _motion[1]
                    '''
                else:
                    self.objects_gt[j].set_data(gt_frame[:, j, 0], gt_frame[:, j, 1])
                if j == 0 or (j > 0 and not self.args.ignore_neighbour_pred):
                    for n in range(self.num_models):
                        if j > 0 and self.args.anim_neighbour_motion:
                            # TODO neighbour motion not working with neighbour predictions
                            _motion = self.pred[n, frame, j] - self.pred[n, frame - 1, j]
                            self.objects_pred[j][n].remove()
                            if torch.all(_motion == 0):
                                self.objects_pred[j] = plt.scatter(self.gt[frame, j, 0], self.gt[frame, j, 1], s=20,
                                                                   color=plt_prediction_colours[n])
                            else:
                                self.objects_pred[j][n] = plt.arrow(
                                    self.gt[n, frame - 1, j, 0], self.gt[n, frame - 1, j, 1], _motion[0], _motion[1],
                                    color=plt_prediction_colours[n], width=0.05, head_width=0.3)
                        else:
                            self.objects_pred[j][n].set_data(pred_frame[n, :, j, 0], pred_frame[n, :, j, 1])
                        if j == 0 and args.plot_pool_shape and models_with_interactions[n] and \
                                self.model_pooling_shapes[n] is not None:
                            if frame_id <= frame:
                                # only draw shape while there is still a portion of prediction left to draw
                                # TODO allow for variable pool shape plotting (for models that have some module from
                                #  simple_shape_config.py or complex_shape_config.py
                                _social_args = model_list[n].interaction_module.args
                                # only allowed for arc shape
                                _arc_radius, _arc_angle = _social_args.arc_radius, _social_args.arc_angle
                                # n_r, n_a = social_args.n_r, social_args.n_a  # not used
                                _arc_position = pred_frame[n, -1, j]
                                _rel_pos = (_arc_position - self.obs[-1, j]) if frame_id == 0 else \
                                    (_arc_position - self.pred[n, frame_id - 1, j])
                                if torch.norm(_rel_pos) == 0:
                                    continue  # if there is no motion, will not update radius
                                _arc_direction = torch.atan2(_rel_pos[1], _rel_pos[0])
                                self.model_pooling_shapes[n].update(_arc_radius, _arc_angle, _arc_position,
                                                                    _arc_direction)
                                shape_updated = True
                            else:
                                # remove the arc shape - not needed anymore
                                s = self.model_pooling_shapes[n]
                                self.model_pooling_shapes[n] = None
                                del s
            if self.args.timestamp_in_plot_title:
                if frame_id > frame:
                    self.plot_title = (self.plot_title_base + ' (full prediction)' if self.plot_title_base else
                                       'Full prediction')
                else:
                    self.plot_title = self.plot_title_base + (f' (t={obs_traj.shape[0] + frame})'
                                                              if self.plot_title_base
                                                              else f'Time t={obs_traj.shape[0] + frame}')
                if self.plot_title:
                    plt.title(self.plot_title, fontdict={'fontsize': 20})
            list_patches_return = []
            if self.args.anim_neighbour_motion:
                for j in range(1, self.num_peds):
                    list_patches_return.append(self.objects_gt[j])
            for s in self.model_pooling_shapes:
                if s is not None and isinstance(s, ArcPatch) and shape_updated:
                    list_patches_return.append(s.arc)  # get modified patch for arc shape
            return list_patches_return

        if x_limits and y_limits:
            plt.xlim(x_limits)
            plt.ylim(y_limits)
        plt.xlabel(f'x ({self.args.units})', fontdict={'fontsize': 16})
        plt.ylabel(f'y ({self.args.units})', fontdict={'fontsize': 16})
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc=self.args.legend_location)
        plt.subplots_adjust(bottom=0.2, left=0.2)  # to approximately center the figure (although with whitespace)
        # blit set to False, cause even though it may improve efficiency, some problems in drawing may occur
        self.anim = FuncAnimation(fig, __update__, frames=num_frames + self.args.maintain_last_instant_for,
                                  init_func=__init_anim__, blit=False, interval=1000)

        AnimatedSituation.num_animations += 1

    def save(self, file_name=None):
        """
        Save the created animation as a video file
        :param file_name: desired file name; should have .mp4 extension; other extensions may not be supported
        :return: nothing is returned from this method
        """
        if file_name is None or not file_name:
            file_name = 'animation.mp4'
        extension_idx = file_name.rfind(".")
        append = str(int(time.time())) if self.args.animation_save_name_append == 'time' \
            else str(AnimatedSituation.num_animations)
        file_name = file_name[:extension_idx] + append + file_name[extension_idx:]
        path = file_name if self.args.animation_parent_dir is None else \
            os.path.join(self.args.animation_parent_dir, file_name)
        # REQUIRES - FFMPEG library
        writer = matplotlib.animation.FFMpegWriter(fps=self.args.fps, extra_args=['-vcodec', 'libx264'])
        self.anim.save(path, writer)


class ArcPatch:
    """
    Class to contain a patch to be drawn in the arc
    """

    def __init__(self, arc_radius, arc_spread, arc_position, arc_direction, ax=None):
        """
        draw the arc (part of a circle) with the lines uniting it from the point of the arc centre position
        :param arc_radius: radius of the arc (in plot units, which usually is metres)
        :param arc_spread: spread or angle of the arc
        :param arc_position: position of the arc to be drawn <-> position of the pedestrian
        :param arc_direction: direction of the arc <-> direction of "gaze", simulated by direction of motion
        :param ax: axes object, to draw the arc patch on.
        """
        theta_start, theta_end = float(- arc_spread / 2 + arc_direction), float(arc_spread / 2 + arc_direction)
        pos_start = arc_position + arc_radius * torch.tensor([np.cos(theta_start), np.sin(theta_start)])
        pos_end = arc_position + arc_radius * torch.tensor([np.cos(theta_end), np.sin(theta_end)])
        self.arc = PltArc(arc_position, arc_radius * 2, arc_radius * 2, theta1=theta_start * 180 / np.pi,
                          theta2=theta_end * 180 / np.pi, linewidth=1, color=pool_shape_color, animated=True)
        if ax is None:
            ax = plt.gca()
        ax.add_patch(self.arc)
        self.ln1, = plt.plot([arc_position[0], pos_start[0]], [arc_position[1], pos_start[1]], linewidth=1,
                             color=pool_shape_color)
        self.ln2, = plt.plot([arc_position[0], pos_end[0]], [arc_position[1], pos_end[1]], linewidth=1,
                             color=pool_shape_color)

    def update(self, arc_radius, arc_spread, arc_position, arc_direction):
        """
        re-draw the arc (part of a circle) with new arguments. Useful for animations
        :param arc_radius: radius of the arc (in plot units, which usually is metres)
        :param arc_spread: spread or angle of the arc
        :param arc_position: position of the arc to be drawn <-> position of the pedestrian
        :param arc_direction: direction of the arc <-> direction of "gaze", simulated by direction of motion
        """
        theta_start, theta_end = float(- arc_spread / 2 + arc_direction), float(arc_spread / 2 + arc_direction)
        pos_start = arc_position + arc_radius * torch.tensor([np.cos(theta_start), np.sin(theta_start)])
        pos_end = arc_position + arc_radius * torch.tensor([np.cos(theta_end), np.sin(theta_end)])
        self.arc.center = arc_position
        self.arc.height = self.arc.width = arc_radius * 2
        self.arc.theta1 = theta_start * 180 / np.pi
        self.arc.theta2 = theta_end * 180 / np.pi
        self.ln1.set_data([arc_position[0], pos_start[0]], [arc_position[1], pos_start[1]])
        self.ln2.set_data([arc_position[0], pos_end[0]], [arc_position[1], pos_end[1]])

    def __del__(self):
        """
        delete the arc, and remove the lines and patches that were drawn
        """
        self.arc.remove()
        self.ln1.remove()
        self.ln2.remove()


def plot_situation(args, obs_traj, gt, predictions, model_labels, x_limits=None, y_limits=None):
    """
    Plot a social situation, which can have multiple pedestrians present
    Assumes Trajnet++ format, with a primary pedestrian and neighbours.
    :param args: command line arguments containing several options to configure the animation
    :param obs_traj: tensor of shape [obs_len, num_peds, 2]. Observed trajectories of all pedestrians in a situation
    :param gt: tensor of shape [pred_len, num_peds, 2]. Ground truth trajectories of all pedestrians in a situation
    :param predictions: tensor of shape [num_models, pred_len, num_peds, 2]. Predicted trajectories of all
    pedestrians in a situation, for num_models supplied (num_models >= 1)
    :param model_labels: list of length num_models, containing labels to identify each of the models
    :param x_limits: limits of the plot along x axis, or None if no limits are meant to be used
    :param y_limits: limits of the plot along y axis, or None if no limits are meant to be used
    :return: nothing, the plot is done via matplotlib.pyplot (plt)
    """
    num_models = len(model_labels)
    # plot primary pedestrian trajectory and corresponding predictions
    obs_prim, gt_prim, preds_prim = obs_traj[:, 0], gt[:, 0], predictions[:, :, 0]
    plt.plot(obs_prim[:, 0], obs_prim[:, 1], linewidth=3, label='OBS')
    plt.scatter(obs_prim[:, 0], obs_prim[:, 1], s=50, alpha=alpha_marker)
    plt.plot(gt_prim[:, 0], gt_prim[:, 1], linewidth=3, label='GT')
    plt.scatter(gt_prim[1:, 0], gt_prim[1:, 1], s=50, alpha=alpha_marker)
    plt_prediction_colours = []
    for i in range(num_models):
        p = plt.plot(preds_prim[i, :, 0], preds_prim[i, :, 1], linewidth=3, label=model_labels[i])
        plt_prediction_colours.append(p[0].get_color())
        plt.scatter(preds_prim[i, 1:, 0], preds_prim[i, 1:, 1], s=20, alpha=alpha_marker)
    # do the same, but for each neighbour
    for n in range(1, obs_traj.shape[1]):
        obs_neigh, gt_neigh, preds_neigh = obs_traj[:, n], gt[:, n], predictions[:, :, n]
        obs_and_gt_neigh = torch.cat((obs_neigh, gt_neigh), dim=0)
        if n == 1:
            plt.plot(obs_and_gt_neigh[:, 0], obs_and_gt_neigh[:, 1], linewidth=2, label='NEIGH', color=neigh_color)
        else:  # no label
            plt.plot(obs_and_gt_neigh[:, 0], obs_and_gt_neigh[:, 1], linewidth=2, color=neigh_color)
        plt.scatter(obs_and_gt_neigh[1:, 0], obs_and_gt_neigh[1:, 1], s=20, alpha=alpha_marker, color=neigh_color)
        if not args.ignore_neighbour_pred:
            for i in range(num_models):
                plt.plot(preds_neigh[i, :, 0], preds_neigh[i, :, 1], linewidth=2, color=plt_prediction_colours[i])
                plt.scatter(preds_neigh[i, 1:, 0], preds_neigh[i, 1:, 1], s=20, alpha=alpha_marker,
                            color=plt_prediction_colours[i])
    if x_limits and y_limits:
        plt.xlim(x_limits)
        plt.ylim(y_limits)
    plt.xlabel(f'x ({args.units})', fontdict={'fontsize': 16})
    plt.ylabel(f'y ({args.units})', fontdict={'fontsize': 16})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    if args.plot_title is not None and args.plot_title:
        plt.title(args.plot_title, fontdict={'fontsize': 20})
    plt.legend()


def apply_plot_args_to_trajectories(args, obs_traj, gt, predictions):
    """
    Apply command line arguments to the trajectories to be plotted
    :param args: command line arguments containing the several changes/restrictions to the trajectories
    :param obs_traj: Tensor of shape [obs_traj_len, num_peds, 2]. Past trajectories of all pedestrians
    :param gt: Tensor of shape [pred_traj_len, num_peds, 2]. Real Future (ground truth) trajectories of all pedestrians
    :param predictions: Tensor of shape [num_methods, pred_traj_len, num_peds, 2]. Predicted trajectories by
    'num_methods' different models, for all pedestrians
    :return: the three aforementioned tensors, with the arguments applied.
    """
    squeeze_unsqueeze = obs_traj.ndim < 3  # for the case of just one pedestrian - shape (traj_len, 2)
    if squeeze_unsqueeze:
        obs_traj, gt, predictions = obs_traj.unsqueeze(1), gt.unsqueeze(1), predictions.unsqueeze(2)
    num_peds = obs_traj.shape[1]
    seq_mask = torch.ones(num_peds, device=obs_traj.device).to(torch.bool)
    if args.ignore_neighbours_past:
        seq_mask *= ~torch.all(torch.isnan(gt[:, :, 0]), dim=0)
    obs_traj, gt, predictions = obs_traj[:, seq_mask], gt[:, seq_mask], predictions[:, :, seq_mask]
    if args.switch_x_y:
        obs_traj[:, :, [0, 1]] = obs_traj[:, :, [1, 0]]
        gt[:, :, [0, 1]] = gt[:, :, [1, 0]]
        predictions[:, :, :, [0, 1]] = predictions[:, :, :, [1, 0]]
    if args.invert_x:
        obs_traj[:, :, 0] = - obs_traj[:, :, 0]
        gt[:, :, 0] = - gt[:, :, 0]
        predictions[:, :, :, 0] = - predictions[:, :, :, 0]
    if args.invert_y:
        obs_traj[:, :, 1] = - obs_traj[:, :, 1]
        gt[:, :, 1] = - gt[:, :, 1]
        predictions[:, :, :, 1] = - predictions[:, :, :, 1]
    if squeeze_unsqueeze:
        return obs_traj.squeeze(1), gt.squeeze(1), predictions.squeeze(2)
    return obs_traj, gt, predictions


def get_or_compute_predictions(model, data_loader_idx, train_args, input_type, output_type, obs_traj_seq, obs_traj,
                               seq_start_end, metadata, pred_traj_len):
    """
    Retrieve pre-computed trajectory predictions for plotting, or compute them here if needed
    :param model: model to compute the predictions, or data loader(s) to load pre-computed predictions
    :param data_loader_idx: in case of multiple data loaders, the index to select the correct one
    :param train_args: argument list used at train time (may have relevant parameters to configure computation)
    :param input_type: type of trajectory that the model expects at input (ABSOLUTE, VELOCITY, ACCELERATION, ETC.)
    :param output_type: type of trajectory that the model outputs (ABSOLUTE, VELOCITY, ACCELERATION, ETC.)
    :param obs_traj_seq: Tensor of shape [obs_len, num_peds, 2]. Observed trajectories, according to input_type
    :param obs_traj: Tensor of shape [obs_len, num_peds, 2]. Observed trajectories, in absolute positions
    :param seq_start_end: Tensor of shape [batch, 2]. To delimit between different situations in the same batch.
    :param metadata: List of size batch. Contains additional metadata to regulate model prediction computations
    :param pred_traj_len: Indicates for how many instants to predict
    :return: Tensor of shape [pred_traj_len, num_peds, 2]. Predicted trajectories for all pedestrians.
    """
    class_name = model.__class__.__name__.lower()
    if 'dataloader' in class_name or (class_name == 'list' and 'dataloader' in model[0].__class__.__name__.lower()):
        # for this model, predictions have been pre-computed and then read from file
        pred_loader = model[data_loader_idx] if isinstance(model, list) else model
        # NOTE THIS DOES NOT SUPPORT MULTIMODALITY (num_samples fixed to 1 and curr_sample equal to the only sample)
        return pred_data_call(pred_loader, obs_traj_seq, pred_traj_len, seq_start_end, metadata, 1, 0)
    model_with_fields = 'fields' in class_name or 'smf' in class_name
    model_with_interactions = 'interaction' in class_name or 'social' in class_name
    # to decide how to call the model forward pass for evaluation
    norm_scene = hasattr(train_args, 'normalize_scene') and train_args.normalize_scene and \
                 input_type == TrajectoryType.ABS
    if norm_scene:
        # directionally normalize the trajectories going in
        obs_traj_seq_in = obs_traj_seq.clone()
        for i, (start, end) in enumerate(seq_start_end):
            obs_traj_seq_in[:, start:end], rot, center = \
                center_scene(obs_traj[:, start:end], obs_length=obs_traj_seq.shape[0])
            metadata[i].angle = rot
            metadata[i].center = center
    else:
        obs_traj_seq_in = obs_traj_seq
    with torch.no_grad():
        if model_with_interactions:
            if model_with_fields:
                prediction = model(obs_traj_seq_in, pred_traj_len, seq_start_end=seq_start_end,
                                   metadata=metadata)
            else:
                prediction = model(obs_traj_seq_in, pred_traj_len, seq_start_end=seq_start_end)
        else:
            prediction = model(obs_traj_seq_in, pred_traj_len)
    if norm_scene:
        # directionally de-normalize the trajectories going in
        if output_type == TrajectoryType.ABS:
            prediction = normalize_sequences(prediction.clone(), seq_start_end, metadata, inverse=True)
        elif output_type == TrajectoryType.VEL:
            # velocity - denormalize absolute trajectory and then denormalize
            abs_pos = relative_traj_to_abs(prediction[:, :, :2].clone(), obs_traj_seq_in[-1])
            prediction_abs = normalize_sequences(abs_pos, seq_start_end, metadata, inverse=True)
            prediction[0, :, :2] = prediction_abs[0] - obs_traj[-1, :, :2]
            prediction[1:, :, :2] = prediction_abs[1:, :, :2] - prediction_abs[:-1, :, :2]
        else:
            raise Exception('NOT IMPLEMENTED - De-normalization with Acceleration')
    return prediction
